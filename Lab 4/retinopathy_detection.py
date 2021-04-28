from dataloader import RetinopathyLoader
from torch import Tensor, device, cuda, no_grad
from torch import max as tensor_max
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from typing import Optional, Type, Union, List, Dict
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import sys
import os
import torch.nn as nn
import torch.optim as op
import torchvision.models as torch_models
import matplotlib.pyplot as plt
import numpy as np


class BasicBlock(nn.Module):
    """
    output = (channels, H, W) -> conv2d (3x3) -> (channels, H, W) -> conv2d (3x3) -> (channels, H, W) + (channels, H, W)
    """
    expansion: int = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, down_sample: Optional[nn.Module] = None):
        super(BasicBlock, self).__init__()

        self.activation = nn.ReLU(inplace=True)
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False),
            nn.BatchNorm2d(out_channels),
            self.activation,
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False),
            nn.BatchNorm2d(out_channels),
        )
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

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, down_sample: Optional[nn.Module] = None):
        super(BottleneckBlock, self).__init__()

        external_channels = out_channels * self.expansion
        self.activation = nn.ReLU(inplace=True)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            self.activation,
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            self.activation,
            nn.Conv2d(in_channels=out_channels,
                      out_channels=external_channels,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(external_channels),
        )
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
            pretrained_resnet = getattr(torch_models, architecture)(pretrained=True)
            self.conv_1 = nn.Sequential(
                getattr(pretrained_resnet, 'conv1'),
                getattr(pretrained_resnet, 'bn1'),
                getattr(pretrained_resnet, 'relu'),
                getattr(pretrained_resnet, 'maxpool')
            )

            # Layers
            self.conv_2 = getattr(pretrained_resnet, 'layer1')
            self.conv_3 = getattr(pretrained_resnet, 'layer2')
            self.conv_4 = getattr(pretrained_resnet, 'layer3')
            self.conv_5 = getattr(pretrained_resnet, 'layer4')

            self.classify = nn.Sequential(
                getattr(pretrained_resnet, 'avgpool'),
                nn.Flatten(),
                nn.Linear(getattr(pretrained_resnet, 'fc').in_features, out_features=50),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(in_features=50, out_features=20),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(in_features=20, out_features=5)
            )

            del pretrained_resnet
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
                nn.Linear(in_features=512 * block.expansion, out_features=50),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(in_features=50, out_features=20),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(in_features=20, out_features=5)
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
            block(in_channels=self.current_channels,
                  out_channels=in_channels,
                  stride=stride,
                  down_sample=down_sample)
        ]
        self.current_channels = in_channels * block.expansion
        layers += [block(in_channels=self.current_channels, out_channels=in_channels) for _ in range(1, num_of_blocks)]

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
    return ResNet(architecture='resnet18', block=BasicBlock, layers=[2, 2, 2, 2], pretrain=pretrain)


def resnet_50(pretrain: bool = False) -> ResNet:
    """
    Get ResNet50
    :param pretrain: whether use pretrained model
    :return: ResNet50
    """
    return ResNet(architecture='resnet50', block=BottleneckBlock, layers=[3, 4, 6, 3], pretrain=pretrain)


def show_results(target_model: str, epochs: int, accuracy: Dict[str, dict], prediction: Dict[str, np.ndarray],
                 ground_truth: np.ndarray, keys: List[str]) -> None:
    """
    Show accuracy results
    :param target_model: ResNet18 or ResNet50
    :param epochs: number of epochs
    :param accuracy: training and testing accuracy of different ResNets
    :param prediction: predictions of different ResNets
    :param ground_truth: ground truth of testing data
    :param keys: names of ResNet w/ or w/o pretraining
    :return: None
    """
    # Get the number of characters of the longest ResNet name
    longest = len(max(keys, key=len)) + 6

    if not os.path.exists('./results'):
        os.mkdir('./results')

    # Plot
    plt.figure(0)
    plt.title(f'Result Comparison ({target_model})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')

    for train_or_test, acc in accuracy.items():
        for model in keys:
            plt.plot(range(epochs), acc[model], label=f'{model}_{train_or_test}')
            spaces = ''.join([' ' for _ in range(longest - len(f'{model}_{train_or_test}'))])
            print(f'{model}_{train_or_test}: {spaces}{max(acc[model]):.2f} %')

    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'./results/{target_model}_comparison.png')
    plt.close()

    for key, pred_labels in prediction.items():
        cm = confusion_matrix(y_true=ground_truth, y_pred=pred_labels, normalize='all')
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3, 4]).plot(cmap=plt.cm.Blues)
        plt.title(f'Normalized confusion matrix ({key})')
        plt.tight_layout()
        plt.savefig(f'./results/{key.replace(" ", "_").replace("/", "_")}_confusion.png')
        plt.close()


def train(target_model: str, batch_size: int, learning_rate: float, epochs: int, optimizer: op, momentum: float,
          weight_decay: float, train_device: device, train_dataset: RetinopathyLoader,
          test_dataset: RetinopathyLoader) -> None:
    """
    Train the models
    :param target_model: ResNet18 or ResNet50
    :param batch_size: batch size
    :param learning_rate: learning rate
    :param epochs: number of epochs
    :param optimizer: optimizer
    :param momentum: momentum for SGD
    :param weight_decay: weight decay factor
    :param train_device: training device (cpu or gpu)
    :param train_dataset: training dataset
    :param test_dataset: testing dataset
    :return: None
    """
    # Setup models w/ or w/o pretraining
    info_log('Setup models ...')
    if target_model == 'ResNet18':
        keys = [
            'ResNet18 (w/o pretraining)',
            'ResNet18 (w/ pretraining)'
        ]
        models = {
            keys[0]: resnet_18().to(train_device),
            keys[1]: resnet_18(pretrain=True).to(train_device)
        }
    else:
        keys = [
            'ResNet50 (w/o pretraining)',
            'ResNet50 (w/ pretraining)'
        ]
        models = {
            keys[0]: resnet_50().to(train_device),
            keys[1]: resnet_50(pretrain=True).to(train_device)
        }

    # Setup accuracy structure
    info_log('Setup accuracy structure ...')
    accuracy = {
        'train': {key: [0 for _ in range(epochs)] for key in keys},
        'test': {key: [0 for _ in range(epochs)] for key in keys}
    }

    # Setup prediction structure
    info_log('Setup prediction structure ...')
    prediction = {
        keys[0]: None,
        keys[1]: None
    }

    # Load data
    info_log('Load data ...')
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    ground_truth = np.array([], dtype=int)
    for _, label in test_loader:
        ground_truth = np.concatenate((ground_truth, label.long().view(-1).numpy()))

    # Start training
    info_log('Start training')
    for key, model in models.items():
        info_log(f'Training {key} ...')
        if optimizer is op.SGD:
            model_optimizer = optimizer(model.parameters(), lr=learning_rate, momentum=momentum,
                                        weight_decay=weight_decay)
        else:
            model_optimizer = optimizer(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        max_test_acc = 0
        for epoch in tqdm(range(epochs)):
            # Train model
            model.train()
            for data, label in train_loader:
                inputs = data.to(train_device)
                labels = label.to(train_device).long().view(-1)

                pred_labels = model.forward(inputs=inputs)

                model_optimizer.zero_grad()
                loss = nn.CrossEntropyLoss()(pred_labels, labels)
                loss.backward()
                model_optimizer.step()

                accuracy['train'][key][epoch] += (tensor_max(pred_labels, 1)[1] == labels).sum().item()
            accuracy['train'][key][epoch] = 100.0 * accuracy['train'][key][epoch] / len(train_dataset)

            # Test model
            model.eval()
            with no_grad():
                pred_labels = np.array([], dtype=int)
                for data, label in test_loader:
                    inputs = data.to(train_device)
                    labels = label.to(train_device).long().view(-1)

                    outputs = model.forward(inputs=inputs)
                    outputs = tensor_max(outputs, 1)[1]
                    pred_labels = np.concatenate((pred_labels, outputs.cpu().numpy()))

                    accuracy['test'][key][epoch] += (outputs == labels).sum().item()
                accuracy['test'][key][epoch] = 100.0 * accuracy['test'][key][epoch] / len(test_dataset)

                if accuracy['test'][key][epoch] > max_test_acc:
                    max_test_acc = accuracy['test'][key][epoch]
                    prediction[key] = pred_labels

            debug_log(f'Train accuracy: {accuracy["train"][key][epoch]:.2f}%')
            debug_log(f'Test accuracy: {accuracy["test"][key][epoch]:.2f}%')
        print()
        cuda.empty_cache()

    # Show results
    show_results(target_model=target_model, epochs=epochs, accuracy=accuracy, prediction=prediction,
                 ground_truth=ground_truth, keys=keys)


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


def check_model_type(input_value: str) -> str:
    """
    Check whether the model is resnet18 or resnet50
    :param input_value: input string value
    :return: model name
    """
    lowercase_input = input_value.lower()
    if lowercase_input != 'resnet18' and lowercase_input != 'resnet50':
        raise ArgumentTypeError('Only "ResNet18" and "ResNet50" are supported.')
    elif lowercase_input == 'resnet18':
        return 'ResNet18'
    else:
        return 'ResNet50'


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
    if int_value != 0 and int_value != 1 and int_value != 2:
        raise ArgumentTypeError(f'Verbosity should be 0, 1 or 2.')
    return int_value


def parse_arguments() -> Namespace:
    """
    Parse arguments
    :return: arguments
    """
    parser = ArgumentParser(description='ResNet')
    parser.add_argument('-t', '--target_model', default='ResNet18', type=check_model_type, help='ResNet18 or ResNet50')
    parser.add_argument('-b', '--batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('-l', '--learning_rate', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('-e', '--epochs', default=10, type=int, help='Number of epochs')
    parser.add_argument('-o', '--optimizer', default='sgd', type=check_optimizer_type, help='Optimizer')
    parser.add_argument('-m', '--momentum', default=0.9, type=float, help='Momentum factor for SGD')
    parser.add_argument('-w', '--weight_decay', default=5e-4, type=float, help='Weight decay (L2 penalty)')
    parser.add_argument('-v', '--verbosity', default=0, type=check_verbosity_type, help='Verbosity level')

    return parser.parse_args()


def main() -> None:
    """
    Main function
    :return: None
    """
    # Parse arguments
    arguments = parse_arguments()
    target_model = arguments.target_model
    batch_size = arguments.batch_size
    learning_rate = arguments.learning_rate
    epochs = arguments.epochs
    optimizer = arguments.optimizer
    momentum = arguments.momentum
    weight_decay = arguments.weight_decay
    global verbosity
    verbosity = arguments.verbosity
    info_log(f'Target model: {target_model}')
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

    # Train models
    train(target_model=target_model,
          batch_size=batch_size,
          learning_rate=learning_rate,
          epochs=epochs,
          optimizer=optimizer,
          momentum=momentum,
          weight_decay=weight_decay,
          train_device=train_device,
          train_dataset=train_dataset,
          test_dataset=test_dataset)


if __name__ == '__main__':
    verbosity = None
    main()
