import numpy as np
import matplotlib.pyplot as plt
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
    def __init__(self, input_links: int, output_links: int, activation: str = 'sigmoid', optimizer: str = 'gd',
                 learning_rate: float = 0.1):
        self.weight = np.random.normal(0, 1, (input_links + 1, output_links))
        self.momentum = np.zeros((input_links + 1, output_links))
        self.sum_of_squares_of_gradients = np.zeros((input_links + 1, output_links))
        self.moving_average_m = np.zeros((input_links + 1, output_links))
        self.moving_average_v = np.zeros((input_links + 1, output_links))
        self.update_times = 1
        self.forward_gradient = None
        self.backward_gradient = None
        self.output = None
        self.activation = activation
        self.optimizer = optimizer
        self.learning_rate = learning_rate

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward feed
        :param inputs: input data for this layer
        :return: outputs computed by this layer
        """
        self.forward_gradient = np.append(inputs, np.ones((inputs.shape[0], 1)), axis=1)
        if self.activation == 'sigmoid':
            self.output = self.sigmoid(np.matmul(self.forward_gradient, self.weight))
        elif self.activation == 'tanh':
            self.output = self.tanh(np.matmul(self.forward_gradient, self.weight))
        elif self.activation == 'relu':
            self.output = self.relu(np.matmul(self.forward_gradient, self.weight))
        elif self.activation == 'leaky_relu':
            self.output = self.leaky_relu(np.matmul(self.forward_gradient, self.weight))
        else:
            # Without activation function
            self.output = np.matmul(self.forward_gradient, self.weight)

        return self.output

    def backward(self, derivative_loss: np.ndarray) -> np.ndarray:
        """
        Backward propagation
        :param derivative_loss: loss from next layer
        :return: loss of this layer
        """
        if self.activation == 'sigmoid':
            self.backward_gradient = np.multiply(self.derivative_sigmoid(self.output), derivative_loss)
        elif self.activation == 'tanh':
            self.backward_gradient = np.multiply(self.derivative_tanh(self.output), derivative_loss)
        elif self.activation == 'relu':
            self.backward_gradient = np.multiply(self.derivative_relu(self.output), derivative_loss)
        elif self.activation == 'leaky_relu':
            self.backward_gradient = np.multiply(self.derivative_leaky_relu(self.output), derivative_loss)
        else:
            # Without activation function
            self.backward_gradient = derivative_loss

        return np.matmul(self.backward_gradient, self.weight[:-1].T)

    def update(self) -> None:
        """
        Update weights
        :return: None
        """
        gradient = np.matmul(self.forward_gradient.T, self.backward_gradient)
        if self.optimizer == 'gd':
            delta_weight = -self.learning_rate * gradient
        elif self.optimizer == 'momentum':
            self.momentum = 0.9 * self.momentum - self.learning_rate * gradient
            delta_weight = self.momentum
        elif self.optimizer == 'adagrad':
            self.sum_of_squares_of_gradients += np.square(gradient)
            delta_weight = -self.learning_rate * gradient / np.sqrt(self.sum_of_squares_of_gradients + 1e-8)
        else:
            # adam
            self.moving_average_m = 0.9 * self.moving_average_m + 0.1 * gradient
            self.moving_average_v = 0.999 * self.moving_average_v + 0.001 * np.square(gradient)
            bias_correction_m = self.moving_average_m / (1.0 - 0.9 ** self.update_times)
            bias_correction_v = self.moving_average_v / (1.0 - 0.999 ** self.update_times)
            self.update_times += 1
            delta_weight = -self.learning_rate * bias_correction_m / (np.sqrt(bias_correction_v) + 1e-8)

        self.weight += delta_weight

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """
        Calculate sigmoid function
        y = 1 / (1 + e^(-x))
        :param x: input data
        :return: sigmoid results
        """
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def derivative_sigmoid(y: np.ndarray) -> np.ndarray:
        """
        Calculate the derivative of sigmoid function
        y' = y(1 - y)
        :param y: value of the sigmoid function
        :return: derivative sigmoid result
        """
        return np.multiply(y, 1.0 - y)

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """
        Calculate tanh function
        y = tanh(x)
        :param x: input data
        :return: tanh results
        """
        return np.tanh(x)

    @staticmethod
    def derivative_tanh(y: np.ndarray) -> np.ndarray:
        """
        Calculate the derivative of tanh function
        y' = 1 - y^2
        :param y: value of the tanh function
        :return: derivative tanh result
        """
        return 1.0 - y ** 2

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """
        Calculate relu function
        y = max(0, x)
        :param x: input data
        :return: relu results
        """
        return np.maximum(0.0, x)

    @staticmethod
    def derivative_relu(y: np.ndarray) -> np.ndarray:
        """
        Calculate the derivative of relu function
        y' = 1 if y > 0
        y' = 0 if y <= 0
        :param y: value of the relu function
        :return: derivative relu result
        """
        return np.heaviside(y, 0.0)

    @staticmethod
    def leaky_relu(x: np.ndarray) -> np.ndarray:
        """
        Calculate leaky relu function
        y = max(0, x) + 0.01 * min(0, x)
        :param x: input data
        :return: relu results
        """
        return np.maximum(0.0, x) + 0.01 * np.minimum(0.0, x)

    @staticmethod
    def derivative_leaky_relu(y: np.ndarray) -> np.ndarray:
        """
        Calculate the derivative of leaky relu function
        y' = 1 if y > 0
        y' = 0.01 if y <= 0
        :param y: value of the relu function
        :return: derivative relu result
        """
        y[y > 0.0] = 1.0
        y[y <= 0.0] = 0.01
        return y


class NeuralNetwork:
    def __init__(self, epoch: int = 1000000, learning_rate: float = 0.1, num_of_layers: int = 2, input_units: int = 2,
                 hidden_units: int = 4, activation: str = 'sigmoid', optimizer: str = 'gd', convolution: bool = False):
        self.num_of_epoch = epoch
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.activation = activation
        self.optimizer = optimizer
        self.convolution = convolution
        self.learning_epoch, self.learning_loss = list(), list()

        # Setup layers
        # Input layer
        self.layers = [Layer(input_units, hidden_units, activation, optimizer, learning_rate)]

        # Hidden layers
        for _ in range(num_of_layers - 1):
            self.layers.append(Layer(hidden_units, hidden_units, activation, optimizer, learning_rate))

        # Output layer
        self.layers.append(Layer(hidden_units, 1, 'sigmoid', optimizer, learning_rate))

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

    def train(self, inputs: np.ndarray, labels: np.ndarray) -> None:
        """
        Train the neural network
        :param inputs: input data
        :param labels: input labels
        :return: None
        """
        for epoch in range(self.num_of_epoch):
            prediction = self.forward(inputs)
            loss = self.mse_loss(prediction=prediction, ground_truth=labels)
            self.backward(self.mse_derivative_loss(prediction=prediction, ground_truth=labels))
            self.update()

            if epoch % 100 == 0:
                print(f'Epoch {epoch} loss : {loss}')
                self.learning_epoch.append(epoch)
                self.learning_loss.append(loss)

            if loss < 0.001:
                break

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Predict the labels of inputs
        :param inputs: input data
        :return: predict labels
        """
        prediction = self.forward(inputs=inputs)
        print(prediction)
        return np.round(prediction)

    def show_result(self, inputs: np.ndarray, labels: np.ndarray) -> None:
        """
        Show the ground truth and predicted results
        :param inputs: input data points
        :param labels: ground truth labels
        :return: None
        """
        # Plot ground truth and prediction
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.title('Ground truth', fontsize=18)
        for idx, point in enumerate(inputs):
            plt.plot(point[0], point[1], 'ro' if labels[idx][0] == 0 else 'bo')

        pred_labels = self.predict(inputs)
        plt.subplot(1, 2, 2)
        plt.title('Predict result', fontsize=18)
        for idx, point in enumerate(inputs):
            plt.plot(point[0], point[1], 'ro' if pred_labels[idx][0] == 0 else 'bo')
        print(f'Activation : {self.activation}')
        print(f'Hidden units : {self.hidden_units}')
        print(f'Optimizer : {self.optimizer}')
        print(f'Accuracy : {float(np.sum(pred_labels == labels)) / len(labels)}')

        # Plot learning curve
        plt.figure()
        plt.title('Learning curve', fontsize=18)
        plt.plot(self.learning_epoch, self.learning_loss)

        plt.show()

    @staticmethod
    def mse_loss(prediction: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
        """
        Mean squared error loss
        :param prediction: prediction from neural network
        :param ground_truth: ground truth
        :return: loss
        """
        return np.mean((prediction - ground_truth) ** 2)

    @staticmethod
    def mse_derivative_loss(prediction: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
        """
        Derivative of MSE loss
        :param prediction: prediction from neural network
        :param ground_truth: ground truth
        :return: derivative loss
        """
        return 2 * (prediction - ground_truth) / len(ground_truth)


def check_data_type(input_value: str) -> int:
    """
    Check whether data type is 0 or 1
    :param input_value: input string value
    :return: integer value
    """
    int_value = int(input_value)
    if int_value != 0 and int_value != 1:
        raise ArgumentTypeError(f'Data type({input_value}) should be 0 or 1.')
    return int_value


def check_activation_type(input_value: str) -> str:
    """
    Check activation function type
    :param input_value: input function type
    :return: original function type if the type is valid
    """
    if input_value != 'none' and input_value != 'sigmoid' and input_value != 'tanh' and input_value != 'relu' and input_value != 'leaky_relu':
        raise ArgumentTypeError(
            f"Activation function type should be 'none' or 'sigmoid' or 'tanh' or 'relu' or 'leaky_relu'.")

    return input_value


def check_optimizer_type(input_value: str) -> str:
    """
    Check optimizer
    :param input_value: input optimizer
    :return: original optimizer if the it is valid
    """
    if input_value != 'gd' and input_value != 'momentum' and input_value != 'adagrad' and input_value != 'adam':
        raise ArgumentTypeError(f"Optimizer should be 'gd', 'momentum', 'adagrad' or 'adam'.")

    return input_value


def parse_arguments() -> Namespace:
    """
    Parse arguments
    :return: all arguments
    """
    parser = ArgumentParser(description='Neural Network')
    parser.add_argument('-d', '--data_type', default=0, type=check_data_type,
                        help='0: linear data points, 1: XOR data points')
    parser.add_argument('-n', '--number_of_data', default=100, type=int, help='Number of data points')
    parser.add_argument('-e', '--epoch', default=1000000, type=int, help='Number of epoch')
    parser.add_argument('-l', '--learning-rate', default=0.1, type=float, help='Learning rate of the neural network')
    parser.add_argument('-u', '--units', default=4, type=int, help='Number of units in each hidden layer')
    parser.add_argument('-a', '--activation', default='sigmoid', type=check_activation_type,
                        help='Type of activation function')
    parser.add_argument('-o', '--optimizer', default='gd', type=check_optimizer_type, help='Type of optimizer')
    parser.add_argument('-c', '--convolution', default=False, type=bool, help='Whether to add convolution layers')

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
    hidden_units = args.units
    activation = args.activation
    optimizer = args.optimizer
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
                                   optimizer=optimizer,
                                   convolution=convolution)
    neural_network.train(inputs=inputs, labels=labels)
    neural_network.show_result(inputs=inputs, labels=labels)


if __name__ == '__main__':
    main()
