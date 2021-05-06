from io import open
from torch import optim, device, Tensor, LongTensor, cat, randn, exp, save, load
from torch.utils.data import Dataset
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from typing import List, Tuple, Dict
from tqdm import tqdm
import random
import torch
import torch.cuda as cuda
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pickle
import json


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
        self.tenses = [
            'simple-present',
            'third-person',
            'present-progressive',
            'simple-past'
        ]

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
        _, next_states = self.lstm(embedded_inputs, (hidden_state, cell_state))
        next_hidden, next_cell = next_states

        # Get hidden mean, log variance, and sampled values
        hidden_mean = self.hidden_mean(next_hidden)
        hidden_log_var = self.hidden_log_var(next_hidden)
        hidden_latent = randn(self.latent_size).to(self.train_device) * exp(0.5 * hidden_log_var).to(
            self.train_device) + hidden_mean

        # Get cell mean and log variance, and sampled values
        cell_mean = self.cell_mean(next_cell)
        cell_log_var = self.cell_log_var(next_cell)
        cell_latent = randn(self.latent_size).to(self.train_device) * exp(0.5 * cell_log_var).to(
            self.train_device) + cell_mean

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

    def forward(self, inputs: LongTensor, hidden_latent: Tensor, cell_latent: Tensor) -> Tuple[Tensor, ...]:
        """
        Forward propagation
        :param inputs: inputs
        :param hidden_latent: hidden latent
        :param cell_latent: cell latent
        :return: output, next hidden latent, next cell hidden latent
        """
        # Embed inputs
        embedded_inputs = self.input_embedding(inputs).view(1, 1, self.hidden_size)

        # Get RNN outputs
        output, next_latents = self.lstm(embedded_inputs, (hidden_latent, cell_latent))
        next_hidden_latent, next_cell_latent = next_latents

        # Get decoded output
        output = self.out(output).view(-1, self.input_size)

        return output, next_hidden_latent, next_cell_latent

    def init_hidden_and_cell(self, hidden_latent: Tensor, cell_latent: Tensor,
                             input_condition: int) -> Tuple[Tensor, ...]:
        """
        Concatenate latent and condition, then convert latent to state and return
        :param hidden_latent: hidden latent
        :param cell_latent: cell latent
        :param input_condition: input condition
        :return: hidden state and cell state
        """
        # Embed condition
        embedded_condition = self.embed_condition(input_condition)

        concatenated_hidden_latent = cat((hidden_latent, embedded_condition), dim=2)
        concatenated_cell_latent = cat((cell_latent, embedded_condition), dim=2)
        return self.hidden_latent_to_hidden_state(concatenated_hidden_latent), self.cell_latent_to_cell_state(
            concatenated_cell_latent)

    def embed_condition(self, condition: int) -> Tensor:
        """
        Embed condition
        :param condition: original condition
        :return: embedded condition
        """
        condition_tensor = LongTensor([condition]).to(self.train_device)
        return self.condition_embedding(condition_tensor).view(1, 1, -1)


def save_model_and_loss_and_score(stored_check_point, losses_and_scores) -> None:
    """
    Save model and losses and score
    :return: None
    """
    if not os.path.exists('./model'):
        os.mkdir('./model')

    highest = None
    if os.path.isfile('./model/highest.json'):
        with open('./model/highest.json', 'r') as f:
            highest = json.load(f)

    max_bleu = max(losses_and_scores['BLEU-4 score'])
    max_epoch = losses_and_scores['BLEU-4 score'].index(max_bleu)
    max_gaussian = losses_and_scores['Gaussian score'][max_epoch]
    if highest:
        if highest['bleu'] > max_bleu or highest['gaussian'] > max_gaussian:
            return
    highest = {
        'bleu': max_bleu,
        'gaussian': max_gaussian
    }
    with open('./model/highest.json', 'w') as f:
        json.dump(highest, f)

    save(stored_check_point, f'./model/CVAE-{max_bleu:.2f}-{max_gaussian:.2f}.pt')
    with open(f'./model/losses_and_scores-{max_bleu:.2f}-{max_gaussian:.2f}.pkl', 'wb') as f:
        pickle.dump(losses_and_scores, f, pickle.HIGHEST_PROTOCOL)


def load_model():
    """
    Load model
    :return: checkpoint
    """
    return load(f'./model/CVAE.pt')


def load_losses_and_scores():
    """
    Load losses and scores
    :return: the stored object
    """
    with open(f'./model/losses_and_scores.pkl', 'rb') as f:
        return pickle.load(f)


def gaussian_score(words: List[List[str]]) -> float:
    """
    Compute Gaussian score
    :param words: words generated by decoder
    :return: score
    """
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
                    debug_log(f'{t}')
                    score += 1
    return score / len(words)


def bleu_score(output: str, reference: str) -> float:
    """
    Compute BLEU-4 score
    :param output: output word
    :param reference: reference word
    :return: BLEU-4 score
    """
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33, 0.33, 0.33)
    else:
        weights = (0.25, 0.25, 0.25, 0.25)
    return sentence_bleu([reference], output, weights=weights, smoothing_function=cc.method1)


def kl_loss(hidden_mean: Tensor, hidden_log_variance: Tensor, cell_mean: Tensor, cell_log_variance: Tensor) -> Tensor:
    """
    Compute KL divergence loss
    :param hidden_mean: mean of hidden state
    :param hidden_log_variance: log variance of hidden state
    :param cell_mean: mean of cell state
    :param cell_log_variance: log variance of cell state
    :return: loss
    """
    return torch.sum(0.5 * (hidden_mean ** 2 + torch.exp(hidden_log_variance) - hidden_log_variance - 1
                            + cell_mean ** 2 + torch.exp(cell_log_variance) - cell_log_variance - 1))


def show_results(epochs: int, losses_and_scores: Dict[str, List[float]]) -> None:
    """
    Show losses and scores
    :param epochs: number of epochs
    :param losses_and_scores: cross entropy loss, KL loss and BLEU-4 score
    :return: None
    """
    if not os.path.exists('./results'):
        os.mkdir('./results')

    # Plot losses
    fig, ax_1 = plt.subplots()
    plt.title('Training loss/ratio curve')
    ax_1.set_xlabel('Epoch')
    ax_1.set_ylabel('Loss')
    ax_2 = ax_1.twinx()
    ax_2.set_ylabel('Score / Weight')

    curve_1, = ax_1.plot(range(epochs), losses_and_scores['KL loss'], label='KL loss')
    curve_2, = ax_1.plot(range(epochs), losses_and_scores['Cross entropy loss'], label='Cross entropy loss')
    curve_3, = ax_2.plot(range(epochs), losses_and_scores['BLEU-4 score'], '.', label='BLEU-4 score')
    curve_4, = ax_2.plot(range(epochs), losses_and_scores['Gaussian score'], '.', label='Gaussian score')
    curve_5, = ax_2.plot(range(epochs), losses_and_scores['Teacher forcing ratio'], '--', label='Teacher forcing ratio')
    curve_6, = ax_2.plot(range(epochs), losses_and_scores['KL weight'], '--', label='KL weight')
    ax_1.legend(handles=[curve_1, curve_2, curve_3, curve_4, curve_5, curve_6], loc='lower right')

    plt.savefig(f'./results/figure.png')
    plt.close()


def decode(input_size: int,
           decoder: DecoderRNN,
           hidden_latent: Tensor,
           cell_latent: Tensor,
           condition: int,
           target_length: int,
           dataset: TenseLoader,
           train_device: device,
           decode_all_inputs: LongTensor = None,
           use_teacher_forcing: int = 0) -> LongTensor:
    """
    Decode and get the output from hidden and cell latent
    :param input_size: input size (word)
    :param decoder: decoder
    :param hidden_latent: hidden latent from encoder
    :param cell_latent: cell latent from encoder
    :param condition: input condition
    :param target_length: target length of the word to be generated by decoder
    :param dataset: training/testing dataset
    :param train_device: training device
    :param decode_all_inputs: ground truth
    :param use_teacher_forcing: Whether use teacher forcing ratio
    :return: outputs
    """
    sos_token = dataset.char_dict.word_to_index['SOS']
    eos_token = dataset.char_dict.word_to_index['EOS']

    hidden_latent, cell_latent = decoder.init_hidden_and_cell(hidden_latent=hidden_latent.view(1, 1, -1),
                                                              cell_latent=cell_latent.view(1, 1, -1),
                                                              input_condition=condition)

    outputs = []
    decode_input = LongTensor([sos_token]).to(train_device)
    for idx in range(target_length):
        decode_input = decode_input.detach()
        output, hidden_latent, cell_latent = decoder.forward(inputs=decode_input,
                                                             hidden_latent=hidden_latent,
                                                             cell_latent=cell_latent)
        outputs.append(output)
        output_one_hot = torch.max(torch.softmax(output, dim=1), 1)[1]
        if output_one_hot.item() == eos_token and not use_teacher_forcing:
            break

        if use_teacher_forcing:
            decode_input = decode_all_inputs[idx + 1: idx + 2]
        else:
            decode_input = output_one_hot
    if len(outputs) != 0:
        outputs = cat(outputs, dim=0)
    else:
        outputs = torch.FloatTensor([]).view(0, input_size).to(train_device)

    return outputs


def generate_word(decoder: DecoderRNN,
                  input_size: int,
                  hidden_noises: Tensor,
                  cell_noises: Tensor,
                  condition: int,
                  target_length: int,
                  test_dataset: TenseLoader,
                  train_device: device) -> str:
    """
    Generate a word based on the given noise and condition
    :param decoder: decoder
    :param input_size: input size (word)
    :param hidden_noises: hidden latent/noise
    :param cell_noises: cell latent/noise
    :param condition: given condition
    :param target_length: target generated word length
    :param test_dataset: testing dataset
    :param train_device: training device
    :return:
    """
    outputs = decode(input_size=input_size,
                     decoder=decoder,
                     hidden_latent=hidden_noises,
                     cell_latent=cell_noises,
                     condition=condition,
                     target_length=target_length,
                     dataset=test_dataset,
                     train_device=train_device)
    outputs_one_hot = torch.max(torch.softmax(outputs, dim=1), 1)[1]
    return test_dataset.char_dict.long_tensor_to_string(outputs_one_hot)


def compute_bleu(encoder: EncoderRNN,
                 decoder: DecoderRNN,
                 input_size: int,
                 test_dataset: TenseLoader,
                 train_device: device) -> float:
    """
    Compute BLEU-4 score for testing
    :param encoder: encoder
    :param decoder: decoder
    :param input_size: input size (word)
    :param test_dataset: testing dataset
    :param train_device: training device
    :return: score
    """
    score = 0.0
    for inputs, input_condition, targets, target_condition in test_dataset:
        inputs = inputs.to(train_device)
        targets = targets.to(train_device)

        hidden, cell = encoder.forward(inputs=inputs[1:],
                                       prev_hidden=encoder.init_hidden_or_cell(),
                                       prev_cell=encoder.init_hidden_or_cell(),
                                       input_condition=input_condition)
        _, _, hidden_latent = hidden
        _, _, cell_latent = cell
        inputs_str = test_dataset.char_dict.long_tensor_to_string(inputs)
        targets_str = test_dataset.char_dict.long_tensor_to_string(targets)
        outputs_str = generate_word(decoder=decoder,
                                    input_size=input_size,
                                    hidden_noises=hidden_latent,
                                    cell_noises=cell_latent,
                                    condition=target_condition,
                                    target_length=targets[:-1].size(0),
                                    test_dataset=test_dataset,
                                    train_device=train_device)
        debug_log(f'Input: {inputs_str}')
        debug_log(f'Target: {targets_str}')
        debug_log(f'Output: {outputs_str}\n')

        # Compute BLEU-4 score
        score += bleu_score(outputs_str, targets_str)
    return score / len(test_dataset)


def compute_gaussian(decoder: DecoderRNN,
                     input_size: int,
                     latent_size: int,
                     test_dataset: TenseLoader,
                     train_device: device) -> float:
    """
    Compute Gaussian score for testing
    :param decoder: decoder
    :param input_size: input size (word)
    :param latent_size: latent size
    :param test_dataset: testing dataset
    :param train_device: training device
    :return: Gaussian score
    """
    words = []
    for _ in range(100):
        hidden_noises, cell_noises = randn(latent_size).to(train_device), randn(latent_size).to(train_device)
        group = []
        for condition in range(len(test_dataset.tenses)):
            group.append(generate_word(decoder=decoder,
                                       input_size=input_size,
                                       hidden_noises=hidden_noises,
                                       cell_noises=cell_noises,
                                       condition=condition,
                                       target_length=28,
                                       test_dataset=test_dataset,
                                       train_device=train_device))
        words.append(group)
    return gaussian_score(words)


def get_current_teacher_forcing_ratio(epoch: int) -> float:
    """
    Get current teacher forcing ratio based on current epoch
    :param epoch: current epoch
    :return: teacher forcing ratio
    """
    if epoch < 150:
        return 1.0

    ratio = 1.0 - 0.005 * (epoch - 150)
    if ratio <= 0.0:
        return ratio
    return ratio


def monotonic_kl_annealing(epoch: int) -> float:
    """
    Get monotonic KL cost annealing based on current epoch
    :param epoch: current epoch
    :return: KL weight
    """
    if epoch < 50:
        return 0.0

    weight = 0.0016 * (epoch - 50)
    if weight >= 1.0:
        return 1.0
    return weight


def cyclical_kl_annealing(epoch: int) -> float:
    """
    Get cyclical KL cost annealing based on current epoch
    :param epoch: current epoch
    :return: KL weight
    """
    if epoch < 50:
        return 0.0

    weight = 0.0032 * ((epoch - 50) % 625)
    if weight >= 1.0:
        return 1.0
    return weight


def train(input_size: int,
          condition_size: int,
          hidden_size: int,
          latent_size: int,
          condition_embedding_size: int,
          kl_weight: float,
          kl_type: str,
          teacher_forcing_ratio: float,
          teacher_type: str,
          learning_rate: float,
          epochs: int,
          load_or_not: int,
          show_only: int,
          train_device: device,
          train_dataset: TenseLoader,
          test_dataset: TenseLoader) -> None:
    """
    Train CVAE
    :param input_size: word size
    :param condition_size: word condition size
    :param hidden_size: hidden size
    :param latent_size: latent size
    :param condition_embedding_size: embedded condition size
    :param kl_weight: KL weight
    :param kl_type: use 'fixed, 'monotonic' or 'cyclical' KL weight
    :param teacher_forcing_ratio: teacher forcing ratio
    :param teacher_type: use 'fixed' or 'decreasing' teacher forcing ratio
    :param learning_rate: learning rate
    :param epochs: number of epochs
    :param load_or_not: whether load the model
    :param show_only: whether only show the stored results
    :param train_device: training device
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :return: None
    """
    # Setup CVAE
    info_log('Setup CVAE ...')
    encoder = EncoderRNN(input_size=input_size,
                         condition_size=condition_size,
                         hidden_size=hidden_size,
                         latent_size=latent_size,
                         condition_embedding_size=condition_embedding_size,
                         train_device=train_device).to(train_device)
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder = DecoderRNN(input_size=input_size,
                         condition_size=condition_size,
                         hidden_size=hidden_size,
                         latent_size=latent_size,
                         condition_embedding_size=condition_embedding_size,
                         train_device=train_device).to(train_device)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    if load_or_not:
        check_point = load_model()
        last_losses_and_scores = load_losses_and_scores()
        encoder.load_state_dict(check_point['encoder_state_dict'])
        encoder_optimizer.load_state_dict(check_point['encoder_optimizer_state_dict'])
        decoder.load_state_dict(check_point['decoder_state_dict'])
        decoder_optimizer.load_state_dict(check_point['decoder_optimizer_state_dict'])
    last_epoch = check_point['epoch'] if load_or_not else 0

    # Losses and scores
    if show_only:
        losses_and_scores = {
            'Cross entropy loss': last_losses_and_scores['Cross entropy loss'][:last_epoch],
            'KL loss': last_losses_and_scores['KL loss'][:last_epoch],
            'BLEU-4 score': last_losses_and_scores['BLEU-4 score'][:last_epoch],
            'Gaussian score': last_losses_and_scores['Gaussian score'][:last_epoch],
            'Teacher forcing ratio': last_losses_and_scores['Teacher forcing ratio'][:last_epoch],
            'KL weight': last_losses_and_scores['KL weight'][:last_epoch]
        }
    elif load_or_not:
        losses_and_scores = {
            'Cross entropy loss': last_losses_and_scores['Cross entropy loss'][:last_epoch] + [0.0 for _ in
                                                                                               range(epochs)],
            'KL loss': last_losses_and_scores['KL loss'][:last_epoch] + [0.0 for _ in range(epochs)],
            'BLEU-4 score': last_losses_and_scores['BLEU-4 score'][:last_epoch] + [0.0 for _ in range(epochs)],
            'Gaussian score': last_losses_and_scores['Gaussian score'][:last_epoch] + [0.0 for _ in range(epochs)],
            'Teacher forcing ratio': last_losses_and_scores['Teacher forcing ratio'][:last_epoch] + [0.0 for _ in
                                                                                                     range(epochs)],
            'KL weight': last_losses_and_scores['KL weight'][:last_epoch] + [0.0 for _ in range(epochs)]
        }
    else:
        losses_and_scores = {
            'Cross entropy loss': [0.0 for _ in range(epochs)],
            'KL loss': [0.0 for _ in range(epochs)],
            'BLEU-4 score': [0.0 for _ in range(epochs)],
            'Gaussian score': [0.0 for _ in range(epochs)],
            'Teacher forcing ratio': [0.0 for _ in range(epochs)],
            'KL weight': [0.0 for _ in range(epochs)]
        }

    # For storing model
    stored_check_point = {
        'epoch': None,
        'encoder_state_dict': None,
        'encoder_optimizer_state_dict': None,
        'decoder_state_dict': None,
        'decoder_optimizer_state_dict': None
    }

    # Start training
    if not show_only:
        info_log('Start training')
        for epoch in tqdm(range(last_epoch, last_epoch + epochs)):
            encoder.train()
            decoder.train()

            # Get current teacher forcing ratio
            if teacher_type == 'fixed':
                losses_and_scores['Teacher forcing ratio'][epoch] = teacher_forcing_ratio
            else:
                losses_and_scores['Teacher forcing ratio'][epoch] = get_current_teacher_forcing_ratio(epoch)

            # Get current KL weight
            if kl_type == 'fixed':
                losses_and_scores['KL weight'][epoch] = kl_weight
            elif kl_type == 'monotonic':
                losses_and_scores['KL weight'][epoch] = monotonic_kl_annealing(epoch)
            else:
                losses_and_scores['KL weight'][epoch] = cyclical_kl_annealing(epoch)

            # Train model
            total_cross_entropy_loss, total_kl_loss = 0, 0
            for inputs, condition in train_dataset:
                inputs = inputs.to(train_device)
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()

                # Encode
                hidden, cell = encoder.forward(inputs=inputs[1:],
                                               prev_hidden=encoder.init_hidden_or_cell(),
                                               prev_cell=encoder.init_hidden_or_cell(),
                                               input_condition=condition)
                hidden_mean, hidden_log_var, hidden_latent = hidden
                cell_mean, cell_log_var, cell_latent = cell

                use_teacher_forcing = True if random.random() < losses_and_scores['Teacher forcing ratio'][
                    epoch] else False

                # Decode
                decode_all_inputs = inputs[:-1]
                outputs = decode(input_size=input_size,
                                 decoder=decoder,
                                 hidden_latent=hidden_latent,
                                 cell_latent=cell_latent,
                                 condition=condition,
                                 target_length=decode_all_inputs.size(0),
                                 dataset=train_dataset,
                                 train_device=train_device,
                                 decode_all_inputs=decode_all_inputs,
                                 use_teacher_forcing=use_teacher_forcing)

                # Backpropagation
                cross_entropy_loss = nn.CrossEntropyLoss(reduction='sum')(outputs, inputs[1:1 + outputs.size(0)])
                kld_loss = kl_loss(hidden_mean=hidden_mean,
                                   hidden_log_variance=hidden_log_var,
                                   cell_mean=cell_mean,
                                   cell_log_variance=cell_log_var)
                (cross_entropy_loss + (losses_and_scores['KL weight'][epoch] * kld_loss)).backward()

                total_cross_entropy_loss += cross_entropy_loss.item()
                total_kl_loss += kld_loss.item()
                if np.isnan(cross_entropy_loss.item()) or np.isnan(kld_loss.item()):
                    raise AttributeError(
                        f'Cross entropy loss: {cross_entropy_loss.item()}, KLD loss: {kld_loss.item()}')

                encoder_optimizer.step()
                decoder_optimizer.step()
            losses_and_scores['Cross entropy loss'][epoch] = total_cross_entropy_loss / len(train_dataset)
            losses_and_scores['KL loss'][epoch] = total_kl_loss / len(train_dataset)

            # Test
            encoder.eval()
            decoder.eval()
            losses_and_scores['BLEU-4 score'][epoch] = compute_bleu(encoder=encoder,
                                                                    decoder=decoder,
                                                                    input_size=input_size,
                                                                    test_dataset=test_dataset,
                                                                    train_device=train_device)
            info_log(f'Average BLEU-4 score: {losses_and_scores["BLEU-4 score"][epoch]:.2f}')

            # Compute Gaussian score
            losses_and_scores['Gaussian score'][epoch] = compute_gaussian(decoder=decoder,
                                                                          input_size=input_size,
                                                                          latent_size=latent_size,
                                                                          test_dataset=test_dataset,
                                                                          train_device=train_device)
            info_log(f'Gaussian score: {losses_and_scores["Gaussian score"][epoch]:.2f}')

            # Save the model and losses and scores
            stored_check_point['epoch'] = epoch + 1
            stored_check_point['encoder_state_dict'] = encoder.state_dict()
            stored_check_point['encoder_optimizer_state_dict'] = encoder_optimizer.state_dict()
            stored_check_point['decoder_state_dict'] = decoder.state_dict()
            stored_check_point['decoder_optimizer_state_dict'] = decoder_optimizer.state_dict()
            save_model_and_loss_and_score(stored_check_point=stored_check_point, losses_and_scores=losses_and_scores)

        show_results(epochs=last_epoch + epochs, losses_and_scores=losses_and_scores)
    else:
        max_bleu = max(losses_and_scores['BLEU-4 score'])
        max_epoch = losses_and_scores['BLEU-4 score'].index(max_bleu)
        max_gaussian = losses_and_scores['Gaussian score'][max_epoch]
        print(f'Max average BLEU-4 score: {max_bleu:.2f}')
        print(f'Max Gaussian score: {max_gaussian:.2f}')
        show_results(epochs=last_epoch, losses_and_scores=losses_and_scores)


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


def check_kl_type(input_value: str) -> str:
    """
    Check whether KL annealing type is fixed, monotonic or cyclical
    :param input_value: input string value
    :return: output string value
    """
    lower_input = input_value.lower()
    if lower_input != 'fixed' and lower_input != 'monotonic' and lower_input != 'cyclical':
        raise ArgumentTypeError(f'KL type should be "fixed", "monotonic" or "cyclical"')
    return lower_input


def check_teacher_type(input_value: str) -> str:
    """
    Check whether teacher forcing type is fixed or decreasing
    :param input_value: input string value
    :return: output string value
    """
    lower_input = input_value.lower()
    if lower_input != 'fixed' and lower_input != 'decreasing':
        raise ArgumentTypeError(f'Teacher forcing type should be "fixed" or "decreasing"')
    return lower_input


def check_load_type(input_value: str) -> int:
    """
    Check whether the load is 0 or 1
    :param input_value: input string value
    :return: integer value
    """
    int_value = int(input_value)
    if int_value != 0 and int_value != 1:
        raise ArgumentTypeError(f'Load should be 0 or 1.')
    return int_value


def check_show_type(input_value: str) -> int:
    """
    Check whether the show_only is 0 or 1
    :param input_value: input string value
    :return: integer value
    """
    int_value = int(input_value)
    if int_value != 0 and int_value != 1:
        raise ArgumentTypeError(f'Show_only should be 0 or 1.')
    return int_value


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
    parser.add_argument('-hs', '--hidden_size', default=256, type=check_hidden_size_type, help='RNN hidden size')
    parser.add_argument('-ls', '--latent_size', default=32, type=int, help='Latent size')
    parser.add_argument('-c', '--condition_embedding_size', default=8, type=int, help='Condition embedding size')
    parser.add_argument('-k', '--kl_weight', default=0.0, type=check_float_type, help='KL weight')
    parser.add_argument('-kt', '--kl_weight_type', default='monotonic', type=check_kl_type,
                        help='Fixed, monotonic or cyclical KL weight')
    parser.add_argument('-t', '--teacher_forcing_ratio', default=0.5, type=check_float_type,
                        help='Teacher forcing ratio')
    parser.add_argument('-tt', '--teacher_forcing_type', default='decreasing', type=check_teacher_type,
                        help='Fixed or decreasing teacher forcing ratio')
    parser.add_argument('-lr', '--learning_rate', default=0.007, type=float, help='Learning rate')
    parser.add_argument('-e', '--epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('-l', '--load', default=0, type=check_load_type,
                        help='Whether load the stored model and accuracies')
    parser.add_argument('-s', '--show_only', default=0, type=check_show_type, help='Whether only show the results')
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
    kl_type = arguments.kl_weight_type
    teacher_forcing_ratio = arguments.teacher_forcing_ratio
    teacher_type = arguments.teacher_forcing_type
    learning_rate = arguments.learning_rate
    epochs = arguments.epochs
    load_or_not = arguments.load
    show_only = arguments.show_only
    if show_only:
        load_or_not = 1
    global verbosity
    verbosity = arguments.verbosity
    info_log(f'Hidden size: {hidden_size}')
    info_log(f'Latent size: {latent_size}')
    info_log(f'Condition embedding size: {condition_embedding_size}')
    info_log(f'KL weight: {kl_weight}')
    info_log(f'KL annealing type: {kl_type}')
    info_log(f'Teacher forcing ratio: {teacher_forcing_ratio}')
    info_log(f'Teacher forcing type: {teacher_type}')
    info_log(f'Learning rate: {learning_rate}')
    info_log(f'Number of epochs: {epochs}')
    info_log(f'Use loaded model: {"True" if load_or_not else "False"}')
    info_log(f'Only show the results: {"True" if show_only else "False"}')

    # Get training device
    train_device = device("cuda" if cuda.is_available() else "cpu")
    info_log(f'Training device: {train_device}')

    # Read data
    info_log('Reading data ...')
    train_dataset = TenseLoader(mode='train')
    test_dataset = TenseLoader(mode='test')

    # Train model
    train(input_size=train_dataset.char_dict.num_of_words,
          condition_size=len(train_dataset.tenses),
          hidden_size=hidden_size,
          latent_size=latent_size,
          condition_embedding_size=condition_embedding_size,
          kl_weight=kl_weight,
          kl_type=kl_type,
          teacher_forcing_ratio=teacher_forcing_ratio,
          teacher_type=teacher_type,
          learning_rate=learning_rate,
          epochs=epochs,
          load_or_not=load_or_not,
          show_only=show_only,
          train_device=train_device,
          train_dataset=train_dataset,
          test_dataset=test_dataset)


if __name__ == '__main__':
    verbosity = None
    main()
