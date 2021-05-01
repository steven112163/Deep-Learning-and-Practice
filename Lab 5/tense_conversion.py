from io import open
from torch import optim, device, Tensor, LongTensor, cat, randn, exp
from torch.utils.data import Dataset
from os import system
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from typing import List, Tuple
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


class CharDict:
    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = {}
        self.num_of_words = 0

        for word in ['SOS', 'EOS']:
            self.add_word(word)

        for num in range(26):
            self.add_word(chr(ord('a') + num))

    def add_word(self, word: str) -> None:
        """
        Add a word to the dictionary
        :param word: word to be added
        :return: None
        """
        if word not in self.word_to_index:
            self.word_to_index[word] = self.num_of_words
            self.index_to_word[self.num_of_words] = word
            self.num_of_words += 1

    def string_to_long_tensor(self, input_string: str) -> LongTensor:
        """
        Convert string to Long Tensor represented by indices
        :param input_string: input string
        :return: Long Tensor represented by indices
        """
        sequence = ['SOS'] + list(input_string) + ['EOS']
        return LongTensor([self.word_to_index[char] for char in sequence])

    def long_tensor_to_string(self, long_tensor: LongTensor) -> str:
        """
        Convert Long Tensor to original string
        :param long_tensor: input Long Tensor
        :return: original string
        """
        original_string = ''
        for obj in long_tensor:
            char = self.index_to_word[obj.item()]
            if len(char) < 2:
                original_string += char
            elif char == 'EOS':
                break
        return original_string


class TenseLoader(Dataset):
    def __init__(self, mode: str):
        if mode == 'train':
            file = './data/train.txt'
        else:
            file = './data/test.txt'

        self.data = np.loadtxt(file, dtype=np.str)

        if mode == 'train':
            self.data = self.data.reshape(-1)
        else:
            # sp, tp, pg, p
            # sp -> p
            # sp -> pg
            # sp -> tp
            # sp -> tp
            # p  -> tp
            # sp -> pg
            # p  -> sp
            # pg -> sp
            # pg -> p
            # pg -> tp
            self.targets = np.array([
                [0, 3],
                [0, 2],
                [0, 1],
                [0, 1],
                [3, 1],
                [0, 2],
                [3, 0],
                [2, 0],
                [2, 3],
                [2, 1],
            ])

        self.char_dict = CharDict()
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        if self.mode == 'train':
            condition = index % 4
            return self.char_dict.string_to_long_tensor(self.data[index]), condition
        else:
            input_long_tensor = self.char_dict.string_to_long_tensor(self.data[index, 0])
            input_condition = self.targets[index, 0]
            output_long_tensor = self.char_dict.string_to_long_tensor(self.data[index, 1])
            output_condition = self.targets[index, 1]
            return input_long_tensor, input_condition, output_long_tensor, output_condition


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

    def forward(self, inputs: LongTensor, prev_hidden: Tensor, prev_cell: Tensor,
                input_condition: int) -> Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]]:
        """
        Forward propagation
        :param inputs: inputs
        :param prev_hidden: previous hidden state
        :param prev_cell: previous cell state
        :param input_condition: input conditions
        :return: (hidden mean, hidden log variance, hidden latent), (cell mean, cell log variance, cell latent)
        """
        # Embed condition
        embedded_condition = self.embed_condition(input_condition)

        # Concatenate previous hidden state with embedded condition to get current hidden state
        hidden_state = cat((prev_hidden, embedded_condition), dim=2)

        # Concatenate previous cell state with embedded condition to get current cell state
        cell_state = cat((prev_cell, embedded_condition), dim=2)

        # Embed inputs
        embedded_inputs = self.input_embedding(inputs).view(-1, 1, self.hidden_size)

        # Get RNN outputs
        _, next_hidden, next_cell = self.lstm(embedded_inputs, (hidden_state, cell_state))

        # Get hidden mean, log variance, and sampled values
        hidden_mean = self.hidden_mean(next_hidden)
        hidden_log_var = self.hidden_log_var(next_hidden)
        hidden_latent = randn(self.latent_size) * exp(0.5 * hidden_log_var) + hidden_mean

        # Get cell mean and log variance, and sampled values
        cell_mean = self.cell_mean(next_cell)
        cell_log_var = self.cell_log_var(next_cell)
        cell_latent = randn(self.latent_size) * exp(0.5 * cell_log_var) + cell_mean

        return (hidden_mean, hidden_log_var, hidden_latent), (cell_mean, cell_log_var, cell_latent)

    def init_hidden_or_cell(self) -> Tensor:
        """
        Return initial hidden or cell state
        :return: Tensor with zeros
        """
        return torch.zeros(1, 1, self.hidden_size - self.condition_embedding_size, device=self.train_device)

    def embed_condition(self, condition: int) -> Tensor:
        """
        Embed condition
        :param condition: original condition
        :return: embedded condition
        """
        condition_tensor = LongTensor([condition]).to(self.train_device)
        return self.condition_embedding(condition_tensor).view(1, 1, -1)


class DecoderRNN(nn.Module):
    def __init__(self, input_size: int, condition_size: int, hidden_size: int, latent_size: int,
                 condition_embedding_size: int, train_device: device):
        super(DecoderRNN, self).__init__()

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

        # Latent to hidden/cell
        self.hidden_latent_to_hidden_state = nn.Linear(in_features=latent_size + condition_embedding_size,
                                                       out_features=hidden_size)
        self.cell_latent_to_cell_state = nn.Linear(in_features=latent_size + condition_embedding_size,
                                                   out_features=hidden_size)

        # RNN
        self.lstm = nn.LSTM(input_size=hidden_size,
                            hidden_size=hidden_size)

        self.out = nn.Linear(hidden_size, input_size)

    def forward(self, inputs: LongTensor, hidden_latent: Tensor, cell_latent: Tensor,
                input_condition: int) -> Tuple[Tensor, ...]:
        """
        Forward propagation
        :param inputs: inputs
        :param hidden_latent: hidden latent
        :param cell_latent: cell latent
        :param input_condition: input conditions
        :return: output, next hidden latent, next cell hidden latent
        """
        # Embed condition
        embedded_condition = self.embed_condition(input_condition)

        # Get hidden state and cell state
        hidden_state, cell_state = self.init_hidden_and_cell(hidden_latent, cell_latent, embedded_condition)

        # Embed inputs
        embedded_inputs = self.input_embedding(inputs).view(1, 1, self.hidden_size)

        # Get RNN outputs
        output, next_hidden_latent, next_cell_latent = self.lstm(embedded_inputs, (hidden_state, cell_state))

        # Get decoded output
        output = self.out(output).view(-1, self.input_size)

        return output, next_hidden_latent, next_cell_latent

    def init_hidden_and_cell(self, hidden_latent: Tensor, cell_latent: Tensor,
                             embedded_condition: Tensor) -> Tuple[Tensor, ...]:
        """
        Concatenate latent and condition, then convert latent to state and return
        :param hidden_latent: hidden latent
        :param cell_latent: cell latent
        :param embedded_condition: embedded condition
        :return: hidden state and cell state
        """
        concatenated_hidden_latent = torch.cat((hidden_latent, embedded_condition), dim=2)
        concatenated_cell_latent = torch.cat((cell_latent, embedded_condition), dim=2)
        return self.hidden_latent_to_hidden_state(concatenated_hidden_latent), self.cell_latent_to_cell_state(
            concatenated_cell_latent)


def gaussian_score(words: List[List[str]]) -> float:
    """
    Compute Gaussian score
    :param words: words?
    :return: score
    """
    # TODO: check words
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
    """
    Compute BLEU-4 score
    :param output:
    :param reference:
    :return: BLEU-4 score
    """
    # TODO: add typing and check output and reference
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
