from io import open
from torch import optim, device, Tensor, LongTensor, cat, randn, exp
from os import system
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from typing import List
import unicodedata
import string
import re
import random
import time
import math
import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as func
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sys


class EncoderRNN(nn.Module):
    def __init__(self, input_size: int, condition_size: int, hidden_size: int, latent_size: int,
                 condition_embedding_size: int, train_device: device):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.condition_size = condition_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.condition_embedding_size = condition_embedding_size
        self.train_device = train_device

        # Embedding
        self.condition_embedding = nn.Embedding(num_embeddings=condition_size,
                                                embedding_dim=condition_embedding_size)
        self.input_embedding = nn.Embedding(num_embeddings=input_size,
                                            embedding_dim=hidden_size)

        # RNN
        self.lstm = nn.LSTM(input_size=hidden_size,
                            hidden_size=hidden_size)

        # Linear layers for hidden state
        self.hidden_mean = nn.Linear(in_features=hidden_size,
                                     out_features=latent_size)
        self.hidden_log_var = nn.Linear(in_features=hidden_size,
                                        out_features=latent_size)

        # Linear layers for cell state
        self.cell_mean = nn.Linear(in_features=hidden_size,
                                   out_features=latent_size)
        self.cell_log_var = nn.Linear(in_features=hidden_size,
                                      out_features=latent_size)

    def forward(self, inputs, prev_hidden, pre_cell, input_condition):
        """
        Forward propagation
        :param inputs: inputs
        :param prev_hidden: previous hidden state
        :param pre_cell: previous cell state
        :param input_condition: input condition
        :return: (hidden mean, hidden log variance, sampled hidden values), (cell mean, cell log variance, sampled cell values)
        """
        # TODO: add typing

        # Embed condition
        embedded_condition = self.embed_condition(input_condition)

        # Concatenate previous hidden state with embedded condition to get current hidden state
        hidden_state = cat((prev_hidden, embedded_condition), dim=2)

        # Concatenate previous cell state with embedded condition to get current cell state
        cell_state = cat((pre_cell, embedded_condition), dim=2)

        # Embed inputs
        embedded_inputs = self.input_embedding(inputs).view(-1, 1, self.hidden_size)

        # Get RNN outputs
        _, next_hidden, next_cell = self.lstm(embedded_inputs, (hidden_state, cell_state))

        # Get hidden mean, log variance, and sampled values
        hidden_mean = self.hidden_mean(next_hidden)
        hidden_log_var = self.hidden_log_var(next_hidden)
        hidden_values = randn(self.latent_size) * exp(0.5 * hidden_log_var) + hidden_mean

        # Get cell mean and log variance, and sampled values
        cell_mean = self.cell_mean(next_cell)
        cell_log_var = self.cell_log_var(next_cell)
        cell_values = randn(self.latent_size) * exp(0.5 * cell_log_var) + cell_mean

        return (hidden_mean, hidden_log_var, hidden_values), (cell_mean, cell_log_var, cell_values)

    def init_hidden_or_cell(self) -> Tensor:
        """
        Return initial hidden or cell state
        :return: Tensor with zeros
        """
        return torch.zeros(1, 1, self.hidden_size - self.condition_embedding_size, device=self.train_device)

    def embed_condition(self, condition) -> Tensor:
        """
        Embed condition
        :param condition: original condition
        :return: embedded condition
        """
        # TODO: add typing
        condition_tensor = LongTensor([condition]).to(self.train_device)
        return self.condition_embedding(condition_tensor).view(1, 1, -1)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, train_device):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.train_device = train_device

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, hidden):
        output = self.embedding(inputs).view(1, 1, -1)
        output = func.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.train_device)


def gaussian_score(words: List[List[str]]) -> float:
    words_list = []
    score = 0
    with open('./data/train.txt', 'r') as fp:
        for line in fp:
            word = line.split(' ')
            word[3] = word[3].strip('\n')
            words_list.extend([word])
        for t in words:
            for i in words_list:
                if t == i:
                    score += 1
    return score / len(words)


def compute_bleu(output, reference) -> float:
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33, 0.33, 0.33)
    else:
        weights = (0.25, 0.25, 0.25, 0.25)
    return sentence_bleu([reference], output, weights=weights, smoothing_function=cc.method1)


def info_log(log: str) -> None:
    """
    Print information log
    :param log: log to be displayed
    :return: None
    """
    global verbosity
    if verbosity:
        print(f'[\033[96mINFO\033[00m] {log}')
        sys.stdout.flush()


def debug_log(log: str) -> None:
    """
    Print debug log
    :param log: log to be displayed
    :return: None
    """
    global verbosity
    if verbosity > 1:
        print(f'[\033[93mDEBUG\033[00m] {log}')
        sys.stdout.flush()


def check_hidden_size_type(input_value: str) -> int:
    """
    Check whether hidden size is 256 or 512
    :param input_value: input string value
    :return: integer value
    """
    int_value = int(input_value)
    if int_value != 256 and int_value != 512:
        raise ArgumentTypeError(f'RNN hidden size should be 256 or 512.')
    return int_value


def check_float_type(input_value: str) -> float:
    """
    Check whether float value is 0 ~ 1
    :param input_value: input string value
    :return: float value
    """
    float_value = float(input_value)
    if float_value < 0.0 or 1.0 < float_value:
        raise ArgumentTypeError(f'Float should be 0 ~ 1.')
    return float_value


def check_verbosity_type(input_value: str) -> int:
    """
    Check whether verbosity is true or false
    :param input_value: input string value
    :return: integer value
    """
    int_value = int(input_value)
    if int_value != 0 and int_value != 1 and int_value != 2:
        raise ArgumentTypeError(f'Verbosity should be 0, 1 or 2.')
    return int_value


def parse_arguments() -> Namespace:
    """
    Parse arguments
    :return: arguments
    """
    parser = ArgumentParser(description='VAE & CVAE')
    parser.add_argument('-h', '--hidden_size', default=256, type=check_hidden_size_type, help='RNN hidden size')
    parser.add_argument('-ls', '--latent_size', default=32, type=int, help='Latent size')
    parser.add_argument('-c', '--condition_embedding_size', default=8, type=int, help='Condition embedding size')
    parser.add_argument('-k', '--kl_weight', default=0.0, type=check_float_type, help='KL weight')
    parser.add_argument('-t', '--teacher_forcing_ratio', default=1.0, type=check_float_type,
                        help='Teacher forcing ratio')
    parser.add_argument('-lr', '--learning_rate', default=0.05, type=float, help='Learning rate')
    parser.add_argument('-v', '--verbosity', default=0, type=check_verbosity_type, help='Verbosity level')

    return parser.parse_args()


def main() -> None:
    """
    Main function
    :return: None
    """
    # Parse arguments
    arguments = parse_arguments()
    hidden_size = arguments.hidden_size
    latent_size = arguments.latent_size
    condition_embedding_size = arguments.condition_embedding_size
    kl_weight = arguments.kl_weight
    teacher_forcing_ratio = arguments.teacher_forcing_ratio
    learning_rate = arguments.learning_rate
    global verbosity
    verbosity = arguments.verbosity
    info_log(f'Hidden size: {hidden_size}')
    info_log(f'Latent size: {latent_size}')
    info_log(f'Condition embedding size: {condition_embedding_size}')
    info_log(f'KL weight: {kl_weight}')
    info_log(f'Teacher forcing ratio: {teacher_forcing_ratio}')
    info_log(f'Learning rate: {learning_rate}')

    # Get training device
    train_device = device("cuda" if cuda.is_available() else "cpu")
    info_log(f'Training device: {train_device}')


if __name__ == '__main__':
    verbosity = None
    main()
