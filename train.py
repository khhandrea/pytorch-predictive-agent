from typing import Any

import numpy as np
from torch import nn, multiprocessing as mp

from agent import PredictiveAgent
from utils import OnPolicyExperienceReplay

def preprocess_observation(observation: np.ndarray,
                        high: np.ndarray,
                        low: np.ndarray
                        ) -> np.ndarray:
    """
    Preprocess numpy array images to specific form. Noralize images between upper bound and lower bound.

    Attributes:
        observation(numpy.ndarray): input images
        high(numpy.ndarray): Upper bound of the pixel value
        low(numpy.ndarray): Lower bound of the pixel value
    """
    result = observation.astype(float) / (high - low) + low
    result = result.astype(np.float32)
    return result

def train(index: int,
          env_class: type,
          env_args: dict[str, Any],
          network_spec: dict[str, Any],
          hyperparameters: dict[str, float],
          queue: mp.Queue,
          global_networks: dict[str, nn.Module]):
    """
    Train function in subprocess. Should be called through torch.multiprocessing.

    Attributes:
        index(int): default process index argument in torch.multiprocessing.spawn
        env_class(type): environment class to be initialized
        env_args(dict[str, Any]): environment attributes for environment to be initialized
        network_spec(dict[str, Any]): torch network specification with dictionary
        hyperparameters(dict[str, float]): training hyperparameters
        queue(torch.multiprocessing.Queue): data structure for torch.multiprocessing
        global_networks(dict[str, torch.nn.Module]): torch.nn.Module networks from main process
    """

    def train_agent():
        train_result = agent.train(batch)
        train_result['reward/extrinsic_return'] = extrinsic_return
        train_result['index'] = index
        train_result['coordinates'] = coordinates
        queue.put(train_result)

    env = env_class(**env_args)
    batch_size = hyperparameters['batch_size']
    agent = PredictiveAgent(env, network_spec, global_networks, hyperparameters)
    replay = OnPolicyExperienceReplay()

    observation, _ = env.reset()
    
    terminated = truncated = False
    batch_step = 0
    extrinsic_return = 0
    coordinates = []
    while not (terminated or truncated):
        observation = preprocess_observation(observation, env.observation_space.high, env.observation_space.low)
        action = agent.get_action(observation)
        next_observation, extrinsic_reward, terminated, truncated, info = env.step(action)
        replay.add_experience(observation, action, extrinsic_reward, terminated or truncated)
        extrinsic_return += extrinsic_reward
        observation = next_observation

        coordinates.append(info['Environment.coordinate'])
        
        batch_step += 1

        if batch_step == batch_size:
            batch = replay.sample(to_tensor=True)
            train_agent()

            extrinsic_return = 0
            coordinates = []
            batch_step = 0
    train_agent()