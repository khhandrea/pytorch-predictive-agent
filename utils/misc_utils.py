from csv import writer
from typing import Iterable
import os

from torch import nn, save
from torch.nn.parameter import Parameter
from torch import optim

def append_to_csv(items: tuple,
                  directory: str,
                  file_name: str
                  ) -> None:
    """
    Append items to end of the csv file. If csv file doesn't exist, create one.

    Attributes:
        items(tuple): data to append. Each item should not be iterable type.
        directory(str): destination directory
        filename(str): destination file name in the directory
    """
    os.makedirs(directory)
    path = os.path.join(directory, file_name)
    with open(path, 'a', newline='') as file:
        writer(file).writerows(items)

def save_module(module: nn.Module,
                directory: str,
                file_name: str
                ) -> None:
    """
    Save a Pytorch module to specific location

    Attributes:
        module(torch.nn.Module): a Pytorch module
        directory(str): destination directory
        file_name(str): destination file name in the directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    save(module.state_dict(),
         os.path.join(directory, file_name))

def initialize_optimizer(optimizer_name: str,
                         params: Iterable[Parameter],
                         learning_rate: float
                         ) -> optim.Optimizer:
    """
    Returns specific optimizer after initialization

    Attributes:
        optimizer_name(str): one of 'sgd', 'adam'
        params(Iterable[torch.nn.parameter.Parameter]): parameters of the module
        learning_rate(float): learning rate of the optimizer

    Returns
        optimizer(torch.optim.Optimmizer): optimizer after initialization
    """
    match optimizer_name:
        case 'sgd':
            optimizer = optim.SGD
        case 'adam':
            optimizer = optim.Adam
        case _:
            raise Exception(f'Invalid optimizer: {optimizer}')
    return optimizer(params, lr=learning_rate)