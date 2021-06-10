from typing import List, Tuple
import matplotlib.pyplot as plt


def plot_losses(losses: Tuple[List[float], ...], labels: List[str], task: str, model: str) -> None:
    """
    Plot losses
    :param losses: losses
    :param labels: label of each loss list
    :param task: task_1 or task_2
    :param model: which model is used
    :return: None
    """
    plt.clf()
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    for idx, loss in enumerate(losses):
        plt.plot(range(len(loss)), loss, label=f'{labels[idx]}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./figure/{task}/{model}_loss.png')


def plot_accuracies(accuracies: Tuple[List[float], ...], labels: List[str], model: str) -> None:
    """
    Plot accuracies
    :param accuracies: accuracies
    :param labels: label of each accuracy list
    :param model: which model is used
    :return: None
    """
    plt.clf()
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    for idx, accuracy in enumerate(accuracies):
        plt.plot(range(len(accuracy)), accuracy, label=labels[idx])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./figure/task_1/{model}_accuracy.png')
