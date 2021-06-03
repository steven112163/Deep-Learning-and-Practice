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
