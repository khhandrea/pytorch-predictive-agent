from typing import Any

from torch import nn, device
from torch import multiprocessing as mp

from agent import PredictiveAgent
from utils import OnPolicyExperienceReplay

def train(
    index: int, # default process index argument in mp.spawn
    env_class: type,
    env_args: dict[str, Any],
    device: device,
    network_spec: dict[str, Any],
    hyperparameters: dict[str, float],
    queue: mp.Queue,
    global_networks: dict[str, nn.Module],
    batch_size: int
):
    env = env_class(**env_args)
    agent = PredictiveAgent(env, device, network_spec, global_networks, hyperparameters)
    replay = OnPolicyExperienceReplay()

    observation, _ = env.reset()
    
    agent.sync_network()
    terminated = truncated = False
    batch_step = 1
    extrinsic_return = 0
    while not (terminated or truncated):
        action = agent.get_action(observation)
        next_observation, extrinsic_reward, terminated, truncated, _ = env.step(action)
        replay.add_experience(observation, action, extrinsic_reward, terminated or truncated)
        extrinsic_return += extrinsic_reward
        observation = next_observation

        if batch_step == batch_size:
            batch = replay.sample()
            batch_result = agent.train(batch)
            batch_result['reward/extrinsic_return'] = extrinsic_return
            queue.put(batch_result)
            extrinsic_return = 0
            batch_step = 0

        batch_step += 1