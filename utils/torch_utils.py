from typing import Iterable
import os

from torch import nn, save
from torch.nn.parameter import Parameter
from torch import optim

def save_module(module: nn.Module,
                directory: str,
                file_name: str
                ) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)
    save(module.state_dict(),
         os.path.join(directory, file_name))

def initialize_optimizer(optimizer_name: str,
                         params: Iterable[Parameter],
                         learning_rate: float
                         ) -> optim.Optimizer:
    match optimizer_name:
        case 'sgd':
            optimizer = optim.SGD
        case 'adam':
            optimizer = optim.Adam
        case _:
            raise Exception(f'Invalid optimizer: {optimizer}')
    return optimizer(params, lr=learning_rate)