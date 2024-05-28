from typing import Any

from cv2 import imwrite
from torch import Tensor, nn, multiprocessing as mp
import gymnasium as gym
import numpy as np

import environments
from agent import PredictiveAgent
from utils import OnPolicyExperienceReplay, preprocess_observation

def train(index: int,
          env_config: dict[str, Any],
          network_spec: dict[str, Any],
          hyperparameters: dict[str, float],
          queue: mp.Queue,
          global_networks: dict[str, nn.Module]):
    """
    Train function in subprocess. Should be called through torch.multiprocessing.

    Attributes:
        index(int): default process index argument in torch.multiprocessing.spawn
        env_config(dict[str, Any]): environment attributes for environment to be initialized with gymnasium.make()
        network_spec(dict[str, Any]): torch network specification with dictionary
        hyperparameters(dict[str, float]): training hyperparameters
        queue(torch.multiprocessing.Queue): data structure for torch.multiprocessing
        global_networks(dict[str, torch.nn.Module]): torch.nn.Module networks from main process
    """
    def train_agent(batch: dict[str, Tensor]):
        train_result = agent.train(batch)
        train_result['reward/extrinsic_return'] = extrinsic_return
        train_result['index'] = index
        train_result['coordinates'] = coordinates
        queue.put(train_result)

    env = gym.make(**env_config)
    batch_size = hyperparameters['batch_size']
    agent = PredictiveAgent(env.action_space.n, network_spec, global_networks, hyperparameters)
    replay = OnPolicyExperienceReplay()

    observation, _ = env.reset()
    
    terminated = truncated = False
    batch_step = 0
    extrinsic_return = 0
    coordinates = []
    while not (terminated or truncated):
        observation = preprocess_observation(observation, (64, 64), 255, 0, 1, -1)
        action = agent.get_action(observation)
        next_observation, extrinsic_reward, terminated, truncated, info = env.step(action)
        replay.add_experience(observation, action, extrinsic_reward, terminated or truncated)
        extrinsic_return += extrinsic_reward
        observation = next_observation

        coordinates.append(info['Environment.coordinate'])
        
        batch_step += 1

        if batch_step == batch_size:
            batch = replay.sample(to_tensor=True)
            train_agent(batch)

            extrinsic_return = 0
            coordinates = []
            batch_step = 0
    if replay.size > 0:
        batch = replay.sample(to_tensor=True)
        train_agent(batch)