from csv import writer
from typing import Iterable
import os

from cv2 import resize
import numpy as np
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
    os.makedirs(directory, exist_ok=True)
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
    os.makedirs(directory, exist_ok=True)
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

def preprocess_observation(observation: np.ndarray,
                           dsize: tuple[int, int],
                           high: float,
                           low: float,
                           ) -> np.ndarray:
    """
    Preprocess numpy array images to specific form. Standardize images between upper bound and lower bound.

    Attributes:
        observation(numpy.ndarray): input images
        high(numpy.ndarray): Upper bound of the pixel value
        low(numpy.ndarray): Lower bound of the pixel value
    """
    # Resize image
    result = np.transpose(observation, (1, 2, 0))
    result = resize(result, dsize)
    result = np.transpose(result, (2, 0, 1))

    # Standardize image
    result = result.astype(float) / (high - low) + low
    result = result.astype(np.float32)
    return result