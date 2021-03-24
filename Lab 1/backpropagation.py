import numpy as np
import matplotlib as plt
from typing import Tuple
from argparse import ArgumentParser, Namespace, ArgumentTypeError


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


def generate_xor_easy(n: int = 11) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate data points based on XOR situation
    :param n: number of points
    :return: inputs and labels
    """
    inputs, labels = [], []

    for i in range(n):
        inputs.append([0.1 * i, 0.1 * i])
        labels.append(0)

        if 0.1 * i == 0.5:
            continue

        inputs.append([0.1 * i, 1 - 0.1 * i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21, 1)


class Layer:
    def __init__(self, input_links: int, output_links: int, learning_rate: int=0.7):
        self.weight = np.random.normal(0, 1, (input_links + 1, output_links))
        self.forward_gradient = None
        self.backward_gradient = None
        self.output = None
        self.learning_rate = learning_rate

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward feed
        :param inputs: input data for this layer
        :return: outputs computed by this layer
        """
        self.forward_gradient = np.append(inputs, np.ones((inputs.shape[0], 1), axis=1))
        self.output = self.sigmoid(np.matmul(self.forward_gradient, self.weight))
        return self.output

    def backward(self, derivative_loss: np.ndarray) -> np.ndarray:
        """
        Backward propagation
        :param derivative_loss: loss from next layer
        :return: loss of this layer
        """
        self.backward_gradient = np.multiply(self.derivative_sigmoid(self.output), derivative_loss)
        return np.matmul(self.backward_gradient, self.weight[:-1].T)

    def update(self) -> None:
        """
        Update weights
        :return: None
        """
        self.weight -= self.learning_rate * np.matmul(self.forward_gradient.T, self.backward_gradient)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """
        Calculate sigmoid function
        :param x: input data
        :return: sigmoid results
        """
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def derivative_sigmoid(y: np.ndarray) -> np.ndarray:
        """
        Calculate the derivative of sigmoid function
        :param y: value of the sigmoid function
        :return: derivative sigmoid result
        """
        return np.multiply(y, 1.0 - y)


class NeuralNetwork:
    def __init__(self, epoch: int = 1000, learning_rate: float = 0.7, hidden_units: int = 4,
                 activation: str = 'sigmoid', convolution: bool = False):
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.activation = activation
        self.convolution = convolution
        self.layers = list()

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward feed
        :param inputs: input data
        :return: predict labels
        """
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, derivative_loss) -> None:
        """
        Backward propagation
        :param derivative_loss: loss form next layer
        :return: None
        """
        for layer in self.layers[::-1]:
            derivative_loss = layer.backward(derivative_loss)

    def update(self) -> None:
        """
        Update all weights in the neural network
        :return: None
        """
        for layer in self.layers:
            layer.update()

    def train(self, inputs: np.ndarray, labels: np.ndarray):
        """
        Train the neural network
        :param inputs: input data
        :param labels: input labels
        :return:
        """

    def predict(self, inputs: np.ndarray):
        """
        Predict the labels of inputs
        :param inputs: input data
        :return: predict labels
        """

    @staticmethod
    def show_result(inputs: np.ndarray, labels: np.ndarray) -> None:
        """
        Show the ground truth and predicted results
        :param inputs: input data points
        :param labels: ground truth labels
        :return: None
        """
        plt.subplot(1, 2, 1)
        plt.title('Ground truth', fontsize=18)
        for idx, point in enumerate(inputs):
            plt.plot(point[0], point[1], 'ro' if labels[idx] == 0 else 'bo')

        # TODO: get pred_labels
        pred_labels = []
        plt.subplot(1, 2, 2)
        plt.title('Predict result', fontsize=18)
        for idx, point in enumerate(inputs):
            plt.plot(point[0], point[1], 'ro' if pred_labels[idx] == 0 else 'bo')

        plt.show()


def check_data_type(input_value: str) -> int:
    """
    Check whether data type is 0 or 1
    :param input_value: input string value
    :return: integer value
    """
    int_value = int(input_value)
    if int_value != 0 or int_value != 1:
        raise ArgumentTypeError(f'Data type({input_value}) should be 0 or 1.')
    return int_value


def parse_arguments() -> Namespace:
    """
    Parse arguments
    :return: all arguments
    """
    parser = ArgumentParser()
    parser.add_argument('--d', '--data_type', default=0, type=check_data_type)
    parser.add_argument('--n', '--number_of_data', default=100, type=int)
    parser.add_argument('--e', '--epoch', default=1000, type=int, help='Number of epoch')
    parser.add_argument('--l', '--learning-rate', default=0.7, type=float, help='Learning rate of the neural network')
    parser.add_argument('--u', '--units', default=4, type=int, help='Number of units in each hidden layer')
    parser.add_argument('--a', '--activation', default='sigmoid', type=str, help='Type of activation function')
    parser.add_argument('--c', '--convolution', default=False, type=bool, help='Whether to add convolution layers')

    return parser.parse_args()


def main() -> None:
    """
    Main function
    :return: None
    """
    args = parse_arguments()
    data_type = args.data_type
    number_of_data = args.number_of_data
    epoch = args.epoch
    learning_rate = args.learning_rate
    hidden_units = args.hidden_units
    activation = args.activation
    convolution = args.convolution

    # Generate data points
    if not data_type:
        inputs, labels = generate_linear(number_of_data)
    else:
        inputs, labels = generate_xor_easy(number_of_data)

    neural_network = NeuralNetwork(epoch=epoch,
                                   learning_rate=learning_rate,
                                   hidden_units=hidden_units,
                                   activation=activation,
                                   convolution=convolution)
    neural_network.train(inputs=inputs, labels=labels)
    neural_network.show_result(inputs=inputs, labels=labels)


if __name__ == '__main__':
    main()
