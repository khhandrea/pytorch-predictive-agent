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

def calc_gaes(rewards, dones, v_preds, gamma, lmbda):
    T = len(rewards)
    assert T + 1 == len(v_preds)
    
    gaes = torch.zeros_like(rewards)
    future_gae = torch.tensor(0.0, dtype=torch.float32)
    not_dones = 1 - dones
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * v_preds[t + 1] * not_dones[t] - v_preds[t]
        gaes[t] = future_gae = delta + gamma * lmbda * future_gae * not_dones[t]
    return gaes