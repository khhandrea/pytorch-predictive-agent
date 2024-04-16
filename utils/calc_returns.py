import torch
from torch import Tensor

def calc_returns(rewards: Tensor,
                 dones: Tensor,
                 last_v_pred: Tensor,
                 gamma: float
                 ) -> Tensor:
    """
    Calculate {batch size}-step TD return.

    Attributes:
        rewards(Tensor): rewards at each time step
        dones(Tensor): mask Tensor whether the episode is end
        last_v_pred(Tensor): state-value prediction at the next step of last step of the reward
        gamma(float): discount factor range of [0, 1]
    """
    assert len(rewards) == len(dones)
    assert len(last_v_pred) == 1

    returns = torch.zeros_like(rewards, dtype=torch.float32)
    not_dones = 1 - dones
    G = last_v_pred
    for t in reversed(range(len(rewards))):
        returns[t] = G = rewards[t] + gamma * G * not_dones[t]
    return returns

def calc_gaes(rewards: Tensor,
              dones: Tensor,
              v_preds: Tensor,
              gamma: float,
              lmbda: float
              ) -> Tensor:
    """
    Calculate general advantage estimation.

    Attributes:
        rewards(Tensor): rewards at each time step
        dones(Tensor): mask Tensor whether the episode is end
        v_preds(Tensor): state-value predictions at each time step
        gamma(float): discount factor range of [0, 1]
        lmbda(float): ratio of the series range of [0, 1]
    """
    T = len(rewards)
    assert T + 1 == len(v_preds)
    
    gaes = torch.zeros_like(rewards)
    future_gae = torch.tensor(0.0, dtype=torch.float32)
    not_dones = 1 - dones
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * v_preds[t + 1] * not_dones[t] - v_preds[t]
        gaes[t] = future_gae = delta + gamma * lmbda * future_gae * not_dones[t]
    return gaes