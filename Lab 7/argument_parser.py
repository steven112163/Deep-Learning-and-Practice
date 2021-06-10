from argparse import ArgumentParser, ArgumentTypeError, Namespace


def check_model_type(input_value: str) -> str:
    """
    Check whether model is gan or nf
    :param input_value: input string value
    :return: string value
    """
    if input_value != 'dcgan' and input_value != 'sagan' and input_value != 'srgan' and input_value != 'glow':
        raise ArgumentTypeError(f'Model should be "dcgan", "sagan", "srgan" or "glow"')
    return input_value.upper()


def parse_arguments() -> Namespace:
    """
    Parse arguments from command line
    :return: arguments
    """
    parser = ArgumentParser(description='cGAN & cNF')
    parser.add_argument('-b', '--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('-i', '--image_size', default=64, type=int, help='Image size')
    parser.add_argument('-w', '--width', default=128, type=int,
                        help='Dimension of the hidden layers in normalizing flow')
    parser.add_argument('-d', '--depth', default=8, type=int, help='Depth of the normalizing flow')
    parser.add_argument('-n', '--num_levels', default=3, type=int, help='Number of levels in normalizing flow')
    parser.add_argument('-gv', '--grad_value_clip', default=0, type=float, help='Clip gradients at specific value')
    parser.add_argument('-gn', '--grad_norm_clip', default=0, type=float, help="Clip gradients' norm at specific value")
    parser.add_argument('-lrd', '--learning_rate_discriminator', default=0.0002, type=float,
                        help='Learning rate of discriminator')
    parser.add_argument('-lrg', '--learning_rate_generator', default=0.0002, type=float,
                        help='Learning rate of generator')
    parser.add_argument('-lrnf', '--learning_rate_normalizing_flow', default=0.0005, type=float,
                        help='Learning rate of normalizing flow')
    parser.add_argument('-e', '--epochs', default=300, type=int, help='Number of epochs')
    parser.add_argument('-wu', '--warmup', default=10, type=int, help='Number of warmup epochs')
    parser.add_argument('-t', '--task', default=1, type=int, choices=[1, 2], help='Task 1 or task 2')
    parser.add_argument('-m', '--model', default='dcgan', type=check_model_type, help='cGAN or cNF')
    parser.add_argument('-inf', '--inference', action='store_true', help='Only infer or not')
    parser.add_argument('-v', '--verbosity', default=0, type=int, choices=[0, 1, 2], help='Verbosity level')

    return parser.parse_args()
