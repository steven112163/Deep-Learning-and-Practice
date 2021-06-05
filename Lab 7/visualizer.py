from typing import List, Tuple
import matplotlib.pyplot as plt


def plot_losses(losses: Tuple[List[float], ...], labels: List[str], task: str, model: str) -> None:
    """
    Plot losses
    :param losses: losses
    :param labels: labels of each loss list
    :param task: task_1 or task_2
    :param model: which model is used
    :return: None
    """
    plt.clf()
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    for idx, loss in enumerate(losses):
        plt.plot(range(len(loss)), loss, label=f'{labels[idx]}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./figure/{task}/{model}_loss.png')


def plot_accuracies(accuracies: List[float], model: str) -> None:
    """
    Plot accuracies
    :param accuracies: accuracies
    :param model: which model is used
    :return: None
    """
    plt.clf()
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.plot(range(len(accuracies)), accuracies, label='accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./figure/task_1/{model}_accuracy.png')
