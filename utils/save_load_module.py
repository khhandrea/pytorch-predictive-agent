import os

import torch

def makedir_and_save_module(state_dict, 
                            path: str,
                            network: str, 
                            description: str):
    path = os.path.join('checkpoints', path, network)
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(
        state_dict,
        os.path.join(path, description) + '.pt'
    )

def get_load_path(load_arg: str, network: str) -> str:
    environment, description, step = load_arg.split('/')
    path = os.path.join('checkpoints', environment, description, network, f'step-{step}') + '.pt'
    return path