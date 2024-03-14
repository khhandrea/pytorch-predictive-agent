from typing import Any

from torch import nn, multiprocessing as mp

from agent import PredictiveAgent

def train(
    env_class: type,
    env_args: dict[str, Any],
    hyperparameter: dict[str, Any],
    queue: mp.Queue,
    global_networks: nn.Module
):
    env = env_class(**env_args)
    agent = PredictiveAgent(**hyperparameter)

    observation, _ = env.reset()
    terminated = truncated = False
    
    step = 1
    extrinsic_reward = 0
    while not (terminated or truncated):
        # action = self._agent.get_action(observation, extrinsic_reward)
        action = env.action_space.sample()
        observation, extrinsic_reward, terminated, truncated, _ = env.step(action)

        step += 1

        if step % 1000 == 0:
            queue.put(f"hello from {step}")