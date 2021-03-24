import numpy as np
import matplotlib as plt
from typing import Tuple
from argparse import ArgumentParser, Namespace


def generate_linear(n: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate data points which are linearly separable
    :param n: number of points
    :return: inputs and labels
    """
    pts = np.random.uniform(0, 1, (n, 2))
    inputs, labels = [], []

    for pt in pts:
        inputs.append([pt[0], pt[1]])
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)

    return np.array(inputs), np.array(labels).reshape(n, 1)


def generate_xor_easy() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate data points based on XOR situation
    :return: inputs and labels
    """
    inputs, labels = [], []

    for i in range(11):
        inputs.append([0.1 * i, 0.1 * i])
        labels.append(0)

        if 0.1 * i == 0.5:
            continue

        inputs.append([0.1 * i, 1 - 0.1 * i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21, 1)


class Layer:
    def __init__(self):
        self.weight = None
        self.output = None


class NeuralNetwork:
    def __init__(self, learning_rate: float = 0.7, hidden_units: int = 4, activation: str = 'sigmoid',
                 convolution: bool = False):
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.activation = activation
        self.convolution = convolution
        self.layers = list()

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """
        Calculate sigmoid function
        :param x: input data
        :return: sigmoid results
        """
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def derivative_sigmoid(x: np.ndarray) -> np.ndarray:
        """
        Calculate the derivative of sigmoid function
        :param x: input data
        :return: derivative sigmoid result
        """
        return np.multiply(x, 1.0 - x)

    @staticmethod
    def show_result(x: np.ndarray, y: np.ndarray, pred_y: np.ndarray) -> None:
        """
        Show the ground truth and predicted results
        :param x: data points
        :param y: ground truth labels
        :param pred_y: predicted labels
        :return: None
        """
        plt.subplot(1, 2, 1)
        plt.title('Ground truth', fontsize=18)
        for idx, point in enumerate(x):
            plt.plot(point[0], point[1], 'ro' if y[idx] == 0 else 'bo')

        plt.subplot(1, 2, 2)
        plt.title('Predict result', fontsize=18)
        for idx, point in enumerate(x):
            plt.plot(point[0], point[1], 'ro' if pred_y[idx] == 0 else 'bo')

        plt.show()


def parse_arguments() -> Namespace:
    """
    Parse arguments
    :return: all arguments
    """
    parser = ArgumentParser()
    parser.add_argument('--l', '--learning-rate')
    parser.add_argument('--u', '--units', type=int)
    parser.add_argument('--a', '--activation', type=int)
    parser.add_argument('--c', '--convolution', default=False, type=bool)

    return parser.parse_args()


def main() -> None:
    """
    Main function
    :return: None
    """
    args = parse_arguments()
    learning_rate = args.learning_rate
    hidden_units = args.hidden_units
    activation = args.activation
    convolution = args.convolution


if __name__ == '__main__':
    main()
