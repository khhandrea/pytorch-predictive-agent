from typing import Any

import torch
from torch import nn, multiprocessing as mp

from agent import PredictiveAgent

def train(
    env_class: type,
    env_args: dict[str, Any],
    device: torch.device,
    hyperparameter: dict[str, Any],
    queue: mp.Queue,
    global_networks: nn.Module,
    batch_size: int
):
    env = env_class(**env_args)
    agent = PredictiveAgent(device, **hyperparameter)

    observation, _ = env.reset()
    terminated = truncated = False
    
    step = 1
    extrinsic_reward = 0
    batch_result = {}
    while not (terminated or truncated):
        action = env.action_space.sample() # action = self._agent.get_action(observation, extrinsic_reward)
        observation, extrinsic_reward, terminated, truncated, _ = env.step(action)
        # append to replay

        if step % batch_size == 0:
            batch = 0 # replay.sample()
            batch_result = agent.train(batch)
            batch_result['reward/extrinsic_return'] = extrinsic_reward
            queue.put(batch_result)

        step += 1