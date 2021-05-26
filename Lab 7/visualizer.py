from typing import List
import matplotlib.pyplot as plt


def plot_losses(generator_losses: List[float], discriminator_losses: List[float]) -> None:
    """
    Plot losses of generator and discriminator
    :param generator_losses: generator losses
    :param discriminator_losses: discriminator losses
    :return: None
    """
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(range(len(generator_losses)), generator_losses, label='Generator')
    plt.plot(range(len(discriminator_losses)), discriminator_losses, label='Discriminator')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./figure/loss.png')
    plt.close()


def plot_accuracies(accuracies: List[float]) -> None:
    """
    Plot accuracies
    :param accuracies: accuracies
    :return: None
    """
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.plot(range(len(accuracies)), accuracies, label='accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./figure/accuracy.png')
    plt.close()
