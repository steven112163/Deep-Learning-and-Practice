from dataloader import RetinopathyLoader
from torch import Tensor, device, cuda, no_grad
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from torchvision.models.utils import load_state_dict_from_url
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from typing import Optional, Type, Union, List
import sys
import torch.nn as nn
import torch.optim as op
import matplotlib.pyplot as plt


class BasicBlock(nn.Module):
    """
    output = (channels, H, W) -> conv2d (3x3) -> (channels, H, W) -> conv2d (3x3) -> (channels, H, W) + (channels, H, W)
    """
    expansion: int = 1

    def __init__(self, channels: int, stride: int = 1, down_sample: Optional[nn.Module] = None):
        super(BasicBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=stride,
                bias=False),
            nn.BatchNorm2d(channels),
            self.activation,
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)
        self.down_sample = down_sample

    def forward(self, inputs: TensorDataset) -> Tensor:
        """
        Forward propagation
        :param inputs: input data
        :return: results
        """
        residual = inputs
        outputs = self.block(inputs)
        if self.down_sample is not None:
            residual = self.down_sample(inputs)

        outputs = self.activation(outputs + residual)

        return outputs


class BottleneckBlock(nn.Module):
    """
    output = (channels * 4, H, W) -> conv2d (1x1) -> (channels, H, W) -> conv2d (3x3) -> (channels, H, W)
             -> conv2d (1x1) -> (channels * 4, H, W) + (channels * 4, H, W)
    """
    expansion: int = 4

    def __init__(self, channels: int, stride: int = 1, down_sample: Optional[nn.Module] = None):
        super(BottleneckBlock, self).__init__()

        external_channels = channels * self.expansion
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=external_channels,
                      out_channels=channels,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(channels),
            self.activation,
            nn.Conv2d(in_channels=channels,
                      out_channels=channels,
                      kernel_size=3,
                      stride=stride,
                      bias=False),
            nn.BatchNorm2d(channels),
            self.activation,
            nn.Conv2d(in_channels=channels,
                      out_channels=external_channels,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(external_channels),
        )
        self.activation = nn.ReLU(inplace=True)
        self.down_sample = down_sample

    def forward(self, inputs: TensorDataset) -> Tensor:
        """
        Forward propagation
        :param inputs: input data
        :return: results
        """
        residual = inputs
        outputs = self.block(inputs)
        if self.down_sample is not None:
            residual = self.down_sample(inputs)

        outputs = self.activation(outputs + residual)

        return outputs


class ResNet(nn.Module):
    def __init__(self, architecture: str, block: Type[Union[BasicBlock, BottleneckBlock]], layers: List[int],
                 pretrain: bool):
        super(ResNet, self).__init__()

        if pretrain:
            model_urls = {
                'resnet_18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
                'resnet_50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
            }

            layers_dict = load_state_dict_from_url(model_urls[architecture])

            self.conv_1 = nn.Sequential(
                layers_dict['conv1'],
                layers_dict['bn1'],
                layers_dict['relu'],
                layers_dict['maxpool']
            )

            # Layers
            self.conv_2 = layers_dict['layer1']
            self.conv_3 = layers_dict['layer2']
            self.conv_4 = layers_dict['layer3']
            self.conv_5 = layers_dict['layer4']

            self.classify = nn.Sequential(
                layers_dict['avgpool'],
                nn.Flatten(),
                nn.Linear(layers_dict['fc'].in_features, out_features=5)
            )

            del layers_dict
        else:
            self.current_channels = 64

            self.conv_1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=64,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3,
                             stride=2,
                             padding=1)
            )

            # Layers
            self.conv_2 = self.make_layer(block=block,
                                          num_of_blocks=layers[0],
                                          in_channels=64)
            self.conv_3 = self.make_layer(block=block,
                                          num_of_blocks=layers[1],
                                          in_channels=128,
                                          stride=2)
            self.conv_4 = self.make_layer(block=block,
                                          num_of_blocks=layers[2],
                                          in_channels=256,
                                          stride=2)
            self.conv_5 = self.make_layer(block=block,
                                          num_of_blocks=layers[3],
                                          in_channels=512,
                                          stride=2)

            self.classify = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(in_features=512 * block.expansion, out_features=5)
            )

    def make_layer(self, block: Type[Union[BasicBlock, BottleneckBlock]], num_of_blocks: int, in_channels: int,
                   stride: int = 1) -> nn.Sequential:
        """
        Make a layer with given block
        :param block: block to be used to compose the layer
        :param num_of_blocks: number of blocks in this layer
        :param in_channels: channels used in the blocks
        :param stride: stride
        :return: convolution layer composed with given block
        """
        down_sample = None
        if stride != 1 or self.current_channels != in_channels * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(in_channels=self.current_channels,
                          out_channels=in_channels * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(in_channels * block.expansion),
            )

        layers = [
            block(channels=in_channels,
                  stride=stride,
                  down_sample=down_sample)
        ]
        self.current_channels = in_channels * block.expansion
        layers += [block(channels=in_channels) for _ in range(1, num_of_blocks)]

        return nn.Sequential(*layers)

    def forward(self, inputs: TensorDataset) -> Tensor:
        """
        Forward propagation
        :param inputs: input data
        :return: results
        """
        partial_results = inputs
        for idx in range(1, 6):
            partial_results = getattr(self, f'conv_{idx}')(partial_results)
        return self.classify(partial_results)


def resnet_18(pretrain: bool = False) -> ResNet:
    """
    Get ResNet18
    :param pretrain: whether use pretrained model
    :return: ResNet18
    """
    return ResNet(architecture='resnet_18', block=BasicBlock, layers=[2, 2, 2, 2], pretrain=pretrain)


def res_net_50(pretrain: bool = False) -> ResNet:
    """
    Get ResNet50
    :param pretrain: whether use pretrained model
    :return: ResNet50
    """
    return ResNet(architecture='resnet_50', block=BottleneckBlock, layers=[3, 4, 6, 3], pretrain=pretrain)


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
