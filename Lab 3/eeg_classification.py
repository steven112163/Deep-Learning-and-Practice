from dataloader import read_bci_data
from torch import Tensor, device, cuda, stack
from torch.utils.data import TensorDataset, DataLoader
from argparse import ArgumentParser, ArgumentTypeError, Namespace
import sys
import torch.nn as nn
import torch.optim as op


class EEGNet(nn.Module):
    def __init__(self, activation: nn.modules.activation):
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

    def forward(self, inputs: TensorDataset) -> Tensor:
        """
        Forward propagation
        :param inputs: input data
        :return: results
        """
        first_conv_results = self.first_conv(inputs)
        depth_wise_conv_results = self.depth_wise_conv(first_conv_results)
        separable_conv_results = self.separable_conv(depth_wise_conv_results)
        return self.classify(separable_conv_results.view(-1, 736))


def train(epochs: int, learning_rate: float, optimizer: op, loss_function: nn.modules.loss, train_device: device,
          train_loader: DataLoader, test_loader: DataLoader, verbosity: int) -> None:
    """
    Train the models
    :param epochs: number of epochs
    :param learning_rate: learning rate
    :param optimizer: optimizer
    :param loss_function: loss function
    :param train_device: training device
    :param train_loader: training data loader
    :param test_loader: testing data loader
    :param verbosity: whether to show info log
    :return: None
    """
    # Setup models for different activation functions
    info_log('Setup models', verbosity=verbosity)
    models = {
        'EEG_ELU': EEGNet(nn.ELU).to(train_device),
        'EEG_ReLU': EEGNet(nn.ReLU).to(train_device),
        'EEG_LeakyReLU': EEGNet(nn.LeakyReLU).to(train_device)
    }

    # Start training
    info_log('Start training', verbosity=verbosity)
    for key, model in models.items():
        info_log(f'Training {key} ...', verbosity=verbosity)
        model_optimizer = optimizer(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            print(f'Epoch: {epoch}')
            # Train model
            for data, label in train_loader:
                inputs = data.to(train_device)
                labels = label.to(train_device).long()

                pred_labels = model.forward(inputs=inputs)

                model_optimizer.zero_grad()
                loss = loss_function(pred_labels, labels)
                print(f'Loss: {loss}')
                loss.backward()
                model_optimizer.step()

    cuda.empty_cache()


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


def check_optimizer_type(input_value: str) -> op:
    """
    Check whether the optimizer is supported
    :param input_value: input string value
    :return: optimizer
    """
    if input_value == 'adam':
        return op.Adam

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


def parse_arguments() -> Namespace:
    """
    Parse arguments
    :return: arguments
    """
    parser = ArgumentParser(description='EEGNet & DeepConvNet')
    parser.add_argument('-e', '--epochs', default=300, type=int, help='Number of epochs')
    parser.add_argument('-lr', '--learning_rate', default=1e-2, type=float, help='Learning rate')
    parser.add_argument('-b', '--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('-o', '--optimizer', default='adam', type=check_optimizer_type, help='Optimizer')
    parser.add_argument('-lf', '--loss_function', default='cross_entropy', type=check_loss_type, help='Loss function')
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
    optimizer = arguments.optimizer
    loss_function = arguments.loss_function
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
          optimizer=optimizer,
          loss_function=loss_function,
          train_device=train_device,
          train_loader=train_loader,
          test_loader=test_loader,
          verbosity=verbosity)


if __name__ == '__main__':
    main()
