from dataloader import read_bci_data
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from torch import device, cuda
from argparse import ArgumentParser, ArgumentTypeError, Namespace
import sys
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, activation=nn.ELU):
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
            nn.Dropout(p=0.25)
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
            nn.Dropout(p=0.25)
        )

        self.classify = nn.Sequential(
            nn.Linear(in_features=736, out_features=2, bias=True)
        )

    def forward(self, inputs: TensorDataset) -> TensorDataset:
        """
        Forward propagation
        :param inputs: input data
        :return: results
        """
        first_conv_results = self.first_conv(inputs)
        depth_wise_conv_results = self.depth_wise_conv(first_conv_results)
        separable_conv_results = self.separable_conv(depth_wise_conv_results)
        flatten_results = separable_conv_results.view(-1, 736)
        results = self.classify(flatten_results)

        return results


def train(epochs: int, learning_rate: float, train_device: device, train_loader: DataLoader, test_loader: DataLoader) -> None:
    """
    Train the models
    :param epochs: number of epochs
    :param learning_rate: learning rate
    :param train_device: training device
    :param train_loader: training data loader
    :param test_loader: testing data loader
    :return: None
    """


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
    parser.add_argument('-e', '--epochs', default=300, type=int, help='Number of epochs')
    parser.add_argument('-l', '--learning_rate', default=1e-2, type=float, help='Learning rate')
    parser.add_argument('-b', '--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('-v', '--verbosity', default=0, type=check_verbosity_type, help='Whether to show info log')

    return parser.parse_args()


def main() -> None:
    """
    Main function
    :return: None
    """
    # Parse arguments
    arguments = parse_arguments()
    epochs = arguments.epochs
    learning_rate = arguments.learning_rate
    batch_size = arguments.batch_size
    verbosity = arguments.verbosity

    # Read data
    info_log('Reading data ...', verbosity=verbosity)
    train_data, train_label, test_data, test_label = read_bci_data()
    train_dataset = TensorDataset(Tensor(train_data), Tensor(train_label))
    test_dataset = TensorDataset(Tensor(test_data), Tensor(test_label))
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, len(test_dataset))

    # Get training device
    train_device = device("cuda" if cuda.is_available() else "cpu")
    info_log(f'Training device: {train_device}', verbosity=verbosity)

    # Train models
    train(epochs=epochs,
          learning_rate=learning_rate,
          train_device=train_device,
          train_loader=train_loader,
          test_loader=test_loader)


if __name__ == '__main__':
    main()
