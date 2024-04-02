from typing import Any

import numpy as np
from torch import nn, multiprocessing as mp

from agent import PredictiveAgent
from utils import OnPolicyExperienceReplay

def preprocess_observation(observation: np.ndarray,
                           high: np.ndarray,
                           low: np.ndarray
                           ) -> np.ndarray:
    result = observation.astype(float) / (high - low) + low
    result = result.astype(np.float32)
    return result

def train(index: int, # default process index argument in mp.spawn
          env_class: type,
          env_args: dict[str, Any],
          network_spec: dict[str, Any],
          hyperparameters: dict[str, float],
          queue: mp.Queue,
          global_networks: dict[str, nn.Module]):
    
    def train_agent():
        train_result = agent.train(batch)
        train_result['reward/extrinsic_return'] = extrinsic_return
        queue.put(train_result)

    env = env_class(**env_args)
    batch_size = hyperparameters['batch_size']
    agent = PredictiveAgent(env, network_spec, global_networks, hyperparameters)
    replay = OnPolicyExperienceReplay()

    observation, _ = env.reset()
    
    terminated = truncated = False
    batch_step = 1
    extrinsic_return = 0
    while not (terminated or truncated):
        observation = preprocess_observation(observation, env.observation_space.high, env.observation_space.low)
        action = agent.get_action(observation)
        next_observation, extrinsic_reward, terminated, truncated, _ = env.step(action)
        replay.add_experience(observation, action, extrinsic_reward, terminated or truncated)
        extrinsic_return += extrinsic_reward
        observation = next_observation
        
        if batch_step == batch_size:
            batch = replay.sample(to_tensor=True)
            train_agent()

            extrinsic_return = 0
            batch_step = 0
        batch_step += 1
    train_agent()