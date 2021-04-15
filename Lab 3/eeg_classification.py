from dataloader import read_bci_data
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from argparse import ArgumentParser, ArgumentTypeError, Namespace
import sys


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


if __name__ == '__main__':
    main()
