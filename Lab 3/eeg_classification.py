from dataloader import read_bci_data
from torch import Tensor, device, cuda, no_grad
from torch import max as tensor_max
from torch.utils.data import TensorDataset, DataLoader
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from typing import Dict, List, Tuple
from functools import reduce
from collections import OrderedDict
import sys
import torch.nn as nn
import torch.optim as op
import matplotlib.pyplot as plt


class EEGNet(nn.Module):
    def __init__(self, activation: nn.modules.activation, dropout: float):
        super().__init__()

        self.first_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(1, 51),
                stride=(1, 1),
                padding=(0, 25),
                bias=False
            ),
            nn.BatchNorm2d(16)
        )

        self.depth_wise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(2, 1),
                stride=(1, 1),
                groups=16,
                bias=False
            ),
            nn.BatchNorm2d(32),
            activation(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=dropout)
        )

        self.separable_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(1, 15),
                stride=(1, 1),
                padding=(0, 7),
                bias=False
            ),
            nn.BatchNorm2d(32),
            activation(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=dropout)
        )

        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=736, out_features=2, bias=True)
        )

    def forward(self, inputs: TensorDataset) -> Tensor:
        """
        Forward propagation
        :param inputs: input data
        :return: results
        """
        first_conv_results = self.first_conv(inputs)
        depth_wise_conv_results = self.depth_wise_conv(first_conv_results)
        separable_conv_results = self.separable_conv(depth_wise_conv_results)
        return self.classify(separable_conv_results)


class DeepConvNet(nn.Module):
    def __init__(self, activation: nn.modules.activation, dropout: float, num_of_linear: int,
                 filters: Tuple[int] = (25, 50, 100, 200)):
        super().__init__()

        self.filters = filters
        self.conv_0 = nn.Sequential(
            # an input = [1, 1, 2, 750]
            nn.Conv2d(
                in_channels=1,
                out_channels=filters[0],
                kernel_size=(1, 5),
                bias=False
            ),
            # an input = [1, 25, 2, 746]
            nn.Conv2d(
                in_channels=filters[0],
                out_channels=filters[0],
                kernel_size=(2, 1),
                bias=False
            ),
            # an input = [1, 25, 1, 746]
            nn.BatchNorm2d(filters[0]),
            activation(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            # an input = [1, 25, 1, 373]
            nn.Dropout(p=dropout)
        )

        for idx, num_of_filters in enumerate(filters[:-1], start=1):
            setattr(self, f'conv_{idx}', nn.Sequential(
                nn.Conv2d(
                    in_channels=num_of_filters,
                    out_channels=filters[idx],
                    kernel_size=(1, 5),
                    bias=False
                ),
                nn.BatchNorm2d(filters[idx]),
                activation(),
                nn.MaxPool2d(kernel_size=(1, 2)),
                nn.Dropout(p=dropout)
            ))

        # If num_of_linear == 1, then there are 2 linear layers
        self.flatten_size = filters[-1] * reduce(lambda x, _: round((x - 4) / 2), filters[:-1], 373)
        interval = round((100.0 - 2.0) / num_of_linear)
        next_layer = 100
        features = [self.flatten_size]
        while next_layer > 2:
            features.append(next_layer)
            next_layer -= interval
        features.append(2)

        layers = [('flatten', nn.Flatten())]
        for idx, in_features in enumerate(features[:-1]):
            layers.append((f'linear_{idx}', nn.Linear(in_features=in_features,
                                                      out_features=features[idx + 1],
                                                      bias=True)))
            if idx != len(features) - 2:
                layers.append((f'activation_{idx}', activation()))
                layers.append((f'dropout_{idx}', nn.Dropout(p=dropout)))
        self.classify = nn.Sequential(OrderedDict(layers))

    def forward(self, inputs: TensorDataset) -> Tensor:
        """
        Forward propagation
        :param inputs: input data
        :return: results
        """
        partial_results = inputs
        for idx in range(len(self.filters)):
            partial_results = getattr(self, f'conv_{idx}')(partial_results)
        return self.classify(partial_results)


def show_results(target_model: str, epochs: int, accuracy: Dict[str, dict], keys: List[str]) -> None:
    """
    Show accuracy results
    :param target_model: target training model
    :param epochs: number of epochs
    :param accuracy: training and testing accuracy of different activation functions
    :param keys: names of neural network with different activation functions
    :return: None
    """
    # Get the number of characters of the longest neural network name
    longest = len(max(keys, key=len)) + 6

    # Plot
    plt.figure(0)
    if target_model == 'EEG':
        plt.title('Activation Function Comparison (EEGNet)')
    else:
        plt.title('Activation Function Comparison (DeepConvNet)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')

    for train_or_test, acc in accuracy.items():
        for model in keys:
            plt.plot(range(epochs), acc[model], label=f'{model}_{train_or_test}')
            spaces = ''.join([' ' for _ in range(longest - len(f'{model}_{train_or_test}'))])
            print(f'{model}_{train_or_test}: {spaces}{max(acc[model]):.2f} %')

    plt.legend(loc='lower right')
    plt.show()


def train(target_model: str, epochs: int, learning_rate: float, batch_size: int, optimizer: op,
          loss_function: nn.modules.loss, dropout: float, num_of_linear: int, train_device: device,
          train_dataset: TensorDataset, test_dataset: TensorDataset, verbosity: int) -> None:
    """
    Train the models
    :param target_model: target training model
    :param epochs: number of epochs
    :param learning_rate: learning rate
    :param batch_size: batch size
    :param optimizer: optimizer
    :param loss_function: loss function
    :param dropout: dropout probability
    :param num_of_linear: number of extra linear layers in DeepConvNet
    :param train_device: training device
    :param train_dataset: training dataset
    :param test_dataset: testing dataset
    :param verbosity: whether to show info log
    :return: None
    """
    # Setup models for different activation functions
    info_log('Setup models', verbosity=verbosity)
    if target_model == 'EEG':
        models = {
            'EEG_ELU': EEGNet(nn.ELU, dropout=dropout).to(train_device),
            'EEG_ReLU': EEGNet(nn.ReLU, dropout=dropout).to(train_device),
            'EEG_LeakyReLU': EEGNet(nn.LeakyReLU, dropout=dropout).to(train_device)
        }
    else:
        models = {
            'Deep_ELU': DeepConvNet(nn.ELU, dropout=dropout, num_of_linear=num_of_linear).to(train_device),
            'Deep_ReLU': DeepConvNet(nn.ReLU, dropout=dropout, num_of_linear=num_of_linear).to(train_device),
            'Deep_LeakyReLU': DeepConvNet(nn.LeakyReLU, dropout=dropout, num_of_linear=num_of_linear).to(train_device)
        }

    # Setup accuracy structure
    info_log('Setup accuracy structure', verbosity=verbosity)
    keys = [f'{target_model}_ELU', f'{target_model}_ReLU', f'{target_model}_LeakyReLU']
    accuracy = {
        'train': {key: [0 for _ in range(epochs)] for key in keys},
        'test': {key: [0 for _ in range(epochs)] for key in keys}
    }

    # Start training
    info_log('Start training', verbosity=verbosity)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, len(test_dataset))
    for key, model in models.items():
        info_log(f'Training {key} ...', verbosity=verbosity)
        model_optimizer = optimizer(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            sys.stdout.write('\r')
            sys.stdout.write(f'Epoch: {epoch + 1} / {epochs}')
            sys.stdout.flush()

            # Train model
            model.train()
            for data, label in train_loader:
                inputs = data.to(train_device)
                labels = label.to(train_device).long()

                pred_labels = model.forward(inputs=inputs)

                model_optimizer.zero_grad()
                loss = loss_function(pred_labels, labels)
                loss.backward()
                model_optimizer.step()

                accuracy['train'][key][epoch] += (tensor_max(pred_labels, 1)[1] == labels).sum().item()
            accuracy['train'][key][epoch] = 100.0 * accuracy['train'][key][epoch] / len(train_dataset)

            # Test model
            model.eval()
            with no_grad():
                for data, label in test_loader:
                    inputs = data.to(train_device)
                    labels = label.to(train_device).long()

                    pred_labels = model.forward(inputs=inputs)

                    accuracy['test'][key][epoch] += (tensor_max(pred_labels, 1)[1] == labels).sum().item()
                accuracy['test'][key][epoch] = 100.0 * accuracy['test'][key][epoch] / len(test_dataset)
        print()
        cuda.empty_cache()

    show_results(target_model=target_model, epochs=epochs, accuracy=accuracy, keys=keys)


def info_log(log: str, verbosity: int) -> None:
    """
    Print information log
    :param log: log to be displayed
    :param verbosity: whether to show info log
    :return: None
    """
    if verbosity:
        print(f'[\033[96mINFO\033[00m] {log}')
        sys.stdout.flush()


def check_model_type(input_value: str) -> op:
    """
    Check whether the model is eeg or deep
    :param input_value: input string value
    :return: model type
    """
    if input_value != 'EEG' and input_value != 'Deep':
        raise ArgumentTypeError(f'Only "EEG" and "Deep" are supported.')

    return input_value


def check_optimizer_type(input_value: str) -> op:
    """
    Check whether the optimizer is supported
    :param input_value: input string value
    :return: optimizer
    """
    if input_value == 'adam':
        return op.Adam
    elif input_value == 'adadelta':
        return op.Adadelta
    elif input_value == 'adagrad':
        return op.Adagrad
    elif input_value == 'adamw':
        return op.AdamW
    elif input_value == 'adamax':
        return op.Adamax

    raise ArgumentTypeError(f'Optimizer {input_value} is not supported.')


def check_loss_type(input_value: str) -> nn.modules.loss:
    """
    Check whether the loss function is supported
    :param input_value: input string value
    :return: loss function
    """
    if input_value == 'cross_entropy':
        return nn.CrossEntropyLoss()

    raise ArgumentTypeError(f'Loss function {input_value} is not supported.')


def check_linear_type(input_value: str) -> int:
    """
    Check whether number of extra linear layers is greater than 0
    :param input_value: input string value
    :return: integer value
    """
    int_value = int(input_value)
    if int_value < 1:
        raise ArgumentTypeError(f'Number of extra linear layers should be greater than 0.')
    return int_value


def check_verbosity_type(input_value: str) -> int:
    """
    Check whether verbosity is true or false
    :param input_value: input string value
    :return: integer value
    """
    int_value = int(input_value)
    if int_value != 0 and int_value != 1:
        raise ArgumentTypeError(f'Verbosity should be 0 or 1.')
    return int_value


def parse_arguments() -> Namespace:
    """
    Parse arguments
    :return: arguments
    """
    parser = ArgumentParser(description='EEGNet & DeepConvNet')
    parser.add_argument('-m', '--model', default='EEG', type=check_model_type, help='EEGNet or DeepConvNet')
    parser.add_argument('-e', '--epochs', default=150, type=int, help='Number of epochs')
    parser.add_argument('-lr', '--learning_rate', default=1e-2, type=float, help='Learning rate')
    parser.add_argument('-b', '--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('-o', '--optimizer', default='adam', type=check_optimizer_type, help='Optimizer')
    parser.add_argument('-lf', '--loss_function', default='cross_entropy', type=check_loss_type, help='Loss function')
    parser.add_argument('-d', '--dropout', default=0.25, type=float, help='Dropout probability')
    parser.add_argument('-l', '--linear', default=1, type=check_linear_type,
                        help='Extra linear layers in DeepConvNet (default is 1)')
    parser.add_argument('-v', '--verbosity', default=0, type=check_verbosity_type, help='Whether to show info log')

    return parser.parse_args()


def main() -> None:
    """
    Main function
    :return: None
    """
    # Parse arguments
    arguments = parse_arguments()
    model = arguments.model
    epochs = arguments.epochs
    learning_rate = arguments.learning_rate
    batch_size = arguments.batch_size
    optimizer = arguments.optimizer
    loss_function = arguments.loss_function
    dropout = arguments.dropout
    num_of_linear = arguments.linear
    verbosity = arguments.verbosity
    info_log(f'Model: {model}', verbosity=verbosity)
    info_log(f'Epochs: {epochs}', verbosity=verbosity)
    info_log(f'Learning rate: {learning_rate}', verbosity=verbosity)
    info_log(f'Batch size: {batch_size}', verbosity=verbosity)
    info_log(f'Optimizer: {optimizer}', verbosity=verbosity)
    info_log(f'Loss function: {loss_function}', verbosity=verbosity)
    info_log(f'Dropout: {dropout}', verbosity=verbosity)
    if model == 'Deep':
        info_log(f'Number of linear layers: {num_of_linear + 1}', verbosity=verbosity)

    # Read data
    info_log('Reading data ...', verbosity=verbosity)
    train_data, train_label, test_data, test_label = read_bci_data()
    train_dataset = TensorDataset(Tensor(train_data), Tensor(train_label))
    test_dataset = TensorDataset(Tensor(test_data), Tensor(test_label))

    # Get training device
    train_device = device("cuda" if cuda.is_available() else "cpu")
    info_log(f'Training device: {train_device}', verbosity=verbosity)

    # Train models
    train(target_model=model,
          epochs=epochs,
          learning_rate=learning_rate,
          batch_size=batch_size,
          optimizer=optimizer,
          loss_function=loss_function,
          dropout=dropout,
          num_of_linear=num_of_linear,
          train_device=train_device,
          train_dataset=train_dataset,
          test_dataset=test_dataset,
          verbosity=verbosity)


if __name__ == '__main__':
    main()
