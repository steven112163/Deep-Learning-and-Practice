import sys
import os


def info_log(log: str, verbosity: int) -> None:
    """
    Print information log
    :param log: log to be displayed
    :param verbosity: verbosity level
    :return: None
    """
    if verbosity:
        print(f'[\033[96mINFO\033[00m] {log}')
        sys.stdout.flush()


def debug_log(log: str, verbosity: int) -> None:
    """
    Print debug log
    :param log: log to be displayed
    :param verbosity: verbosity level
    :return: None
    """
    if verbosity > 1:
        print(f'[\033[93mDEBUG\033[00m] {log}')
        sys.stdout.flush()


def create_directories() -> None:
    """
    Create all directories needed in this lab
    :return: None
    """
    if not os.path.exists('./model/task_1'):
        os.makedirs('./model/task_1')
    if not os.path.exists('./model/task_2'):
        os.makedirs('./model/task_2')
    if not os.path.exists('./test_figure'):
        os.mkdir('./test_figure')
    if not os.path.exists('./figure/task_1'):
        os.makedirs('./figure/task_1')
    if not os.path.exists('./figure/task_2'):
        os.makedirs('./figure/task_2')


def get_score(test: float, new_test: float) -> float:
    """
    Get score according to the test and new_test accuracy
    :param test: Test accuracy
    :param new_test: New test accuracy
    :return: Score
    """
    score = 0.0
    if test >= 0.8:
        score += 0.05 * 100
    elif 0.8 > test >= 0.7:
        score += 0.05 * 90
    elif 0.7 > test >= 0.6:
        score += 0.05 * 80
    elif 0.6 > test >= 0.5:
        score += 0.05 * 70
    elif 0.5 > test >= 0.4:
        score += 0.05 * 60

    if new_test >= 0.8:
        score += 0.1 * 100
    elif 0.8 > new_test >= 0.7:
        score += 0.1 * 90
    elif 0.7 > new_test >= 0.6:
        score += 0.1 * 80
    elif 0.6 > new_test >= 0.5:
        score += 0.1 * 70
    elif 0.5 > new_test >= 0.4:
        score += 0.1 * 60

    return score
