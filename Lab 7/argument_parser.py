from argparse import ArgumentParser, ArgumentTypeError, Namespace


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
    parser.add_argument('-b', '--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('-i', '--image_size', default=64, type=int, help='Image size')
    parser.add_argument('-lrd', '--learning_rate_discriminator', default=0.001, type=float,
                        help='Learning rate of discriminator')
    parser.add_argument('-lrg', '--learning_rate_generator', default=0.001, type=float,
                        help='Learning rate of generator')
    parser.add_argument('-e', '--epochs', default=800, type=int, help='Number of epochs')
    parser.add_argument('-v', '--verbosity', default=0, type=check_verbosity_type, help='Verbosity level')

    return parser.parse_args()
