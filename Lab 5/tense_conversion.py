from io import open
from torch import optim, device
from os import system
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from argparse import ArgumentParser, ArgumentTypeError, Namespace
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
    def __init__(self, input_size, hidden_size, train_device):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.train_device = train_device

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, inputs, hidden):
        embedded = self.embedding(inputs).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.train_device)


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
