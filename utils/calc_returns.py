import torch
from torch import Tensor

def calc_returns(rewards: Tensor,
                 dones: Tensor,
                 last_v_pred: Tensor,
                 gamma: float
                 ) -> Tensor:
    assert len(rewards) == len(dones)
    assert len(last_v_pred) == 1

    returns = torch.zeros_like(rewards, dtype=torch.float32)
    not_dones = 1 - dones
    G = last_v_pred
    for t in reversed(range(len(rewards))):
        returns[t] = G = rewards[t] + gamma * G * not_dones[t]
    return returns