from typing import List, Tuple
import matplotlib.pyplot as plt


def plot_losses(losses: Tuple[List[float], ...], labels: List[str], epoch:int, task: str, model: str) -> None:
    """
    Plot losses
    :param losses: Losses
    :param labels: Label of each loss list
    :param epoch: Current epoch
    :param task: Task_1 or task_2
    :param model: Which model is used
    :return: None
    """
    plt.clf()
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    for idx, loss in enumerate(losses):
        plt.plot(range(epoch+1), loss[:epoch+1], label=f'{labels[idx]}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./figure/{task}/{model}_loss.png')


def plot_accuracies(accuracies: Tuple[List[float], ...], labels: List[str], epoch: int, model: str) -> None:
    """
    Plot accuracies
    :param accuracies: Accuracies
    :param labels: Label of each accuracy list
    :param epoch: Current epoch
    :param model: Which model is used
    :return: None
    """
    plt.clf()
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    for idx, accuracy in enumerate(accuracies):
        plt.plot(range(epoch + 1), accuracy[:epoch + 1], label=labels[idx])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./figure/task_1/{model}_accuracy.png')
