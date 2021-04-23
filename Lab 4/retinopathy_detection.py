from dataloader import RetinopathyLoader
from torch import Tensor, device, cuda, no_grad
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from argparse import ArgumentParser, ArgumentTypeError, Namespace
import sys
import torch.nn as nn
import torch.optim as op
import matplotlib.pyplot as plt


class BasicBlock(nn.Module):
    """
    output = (channels, H, W) -> conv2d (3x3) -> (channels, H, W) -> conv2d (3x3) -> (channels, H, W) + (channels, H, W)
    """
    def __init__(self, channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            self.activation,
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                bias=False
            ),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: TensorDataset) -> Tensor:
        """
        Forward propagation
        :param inputs: input data
        :return: results
        """
        residual = inputs
        outputs = self.block(inputs)
        outputs = self.activation(outputs + residual)

        return outputs


class BottleneckBlock(nn.Module):
    """
    output = (channels, H, W) -> conv2d (1x1) -> (channels / 4, H, W) -> conv2d (3x3) -> (channels / 4, H, W) -> conv2d (1x1) -> (channels, H, W) + (channels, H, W)
    """
    def __init__(self, channels: int):
        super().__init__()

        internal_channels = int(channels / 4.0)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=channels,
                      out_channels=internal_channels,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(internal_channels),
            self.activation,
            nn.Conv2d(in_channels=internal_channels,
                      out_channels=internal_channels,
                      kernel_size=3,
                      bias=False),
            nn.BatchNorm2d(internal_channels),
            self.activation,
            nn.Conv2d(in_channels=internal_channels,
                      out_channels=channels,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: TensorDataset) -> Tensor:
        """
        Forward propagation
        :param inputs: input data
        :return: results
        """
        residual = inputs
        outputs = self.block(inputs)
        outputs = self.activation(outputs + residual)

        return outputs


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


def check_optimizer_type(input_value: str) -> op:
    """
    Check whether the optimizer is supported
    :param input_value: input string value
    :return: optimizer
    """
    if input_value == 'sgd':
        return op.SGD
    elif input_value == 'adam':
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
    parser = ArgumentParser(description='ResNet')
    parser.add_argument('-b', '--batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('-l', '--learning_rate', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('-e', '--epochs', default=10, type=int, help='Number of epochs')
    parser.add_argument('-o', '--optimizer', default='sgd', type=check_optimizer_type, help='Optimizer')
    parser.add_argument('-m', '--momentum', default=0.9, type=float, help='Momentum factor for SGD')
    parser.add_argument('-w', '--weight_decay', default=5e-4, type=float, help='Weight decay (L2 penalty)')
    parser.add_argument('-v', '--verbosity', default=0, type=check_verbosity_type, help='Whether to show info log')

    return parser.parse_args()


def main() -> None:
    """
    Main function
    :return: None
    """
    # Parse arguments
    arguments = parse_arguments()
    batch_size = arguments.batch_size
    learning_rate = arguments.learning_rate
    epochs = arguments.epochs
    optimizer = arguments.optimizer
    momentum = arguments.momentum
    weight_decay = arguments.weight_decay
    global verbosity
    verbosity = arguments.verbosity
    info_log(f'Batch size: {batch_size}')
    info_log(f'Learning rate: {learning_rate}')
    info_log(f'Epochs: {epochs}')
    info_log(f'Optimizer: {optimizer}')
    info_log(f'Momentum: {momentum}')
    info_log(f'Weight decay: {weight_decay}')

    # Read data
    info_log('Reading data ...')
    train_dataset = RetinopathyLoader('./data', 'train', [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ])
    test_dataset = RetinopathyLoader('./data', 'test')

    # Get training device
    train_device = device("cuda" if cuda.is_available() else "cpu")
    info_log(f'Training device: {train_device}')


if __name__ == '__main__':
    verbosity = None
    main()
