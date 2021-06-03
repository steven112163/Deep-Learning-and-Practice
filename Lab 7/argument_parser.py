from argparse import ArgumentParser, ArgumentTypeError, Namespace


def check_task_type(input_value: str) -> int:
    """
    Check whether task is true or false
    :param input_value: input string value
    :return: integer value
    """
    int_value = int(input_value)
    if int_value != 1 and int_value != 2:
        raise ArgumentTypeError(f'Task should be 1 or 2.')
    return int_value


def check_model_type(input_value: str) -> str:
    """
    Check whether model is gan or nf
    :param input_value: input string value
    :return: string value
    """
    if input_value != 'dcgan' and input_value != 'sagan' and input_value != 'srgan' and input_value != 'glow':
        raise ArgumentTypeError(f'Model should be "dcgan", "sagan", "srgan" or "glow"')
    return input_value.upper()

def check_inference_type(input_value: str) -> int:
    """
    Check whether inference is true or false
    :param input_value: input string value
    :return: integer value
    """
    int_value = int(input_value)
    if int_value != 0 and int_value != 1:
        raise ArgumentTypeError(f'Inference should be 0 or 1.')
    return int_value


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
    Parse arguments from command line
    :return: arguments
    """
    parser = ArgumentParser(description='cGAN & cNF')
    parser.add_argument('-b', '--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('-i', '--image_size', default=64, type=int, help='Image size')
    parser.add_argument('-w', '--width', default=64, type=int,
                        help='Dimension of the hidden layers in normalizing flow')
    parser.add_argument('-d', '--depth', default=16, type=int, help='Depth of the normalizing flow')
    parser.add_argument('-n', '--num_levels', default=3, type=int, help='Number of levels in normalizing flow')
    parser.add_argument('-g', '--grad_norm_clip', default=50, type=float, help='Clip gradients during training')
    parser.add_argument('-lrd', '--learning_rate_discriminator', default=0.0005, type=float,
                        help='Learning rate of discriminator')
    parser.add_argument('-lrg', '--learning_rate_generator', default=0.0001, type=float,
                        help='Learning rate of generator')
    parser.add_argument('-lrnf', '--learning_rate_normalizing_flow', default=0.001, type=float,
                        help='Learning rate of normalizing flow')
    parser.add_argument('-e', '--epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('-t', '--task', default=1, type=check_task_type, help='Task 1 or task 2')
    parser.add_argument('-m', '--model', default='dcgan', type=check_model_type, help='cGAN or cNF')
    parser.add_argument('-inf', '--inference', default=0, type=check_inference_type, help='Only infer or not')
    parser.add_argument('-v', '--verbosity', default=0, type=check_verbosity_type, help='Verbosity level')

    return parser.parse_args()
