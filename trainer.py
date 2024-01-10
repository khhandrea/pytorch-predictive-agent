from typing import Dict, Any

from gymnasium.utils.env_checker import check_env
from gymnasium import Env
import numpy as np

from agent import PredictAgent
from utils import LogWriter

class Trainer:
    def __init__(self, env: Env, config: Dict[str, Any]):
        self._env = env
        self._random_policy = config['random_policy']
        self._agent = PredictAgent(
            action_space=self._env.action_space, 
            random_policy=self._random_policy)
        self._log_writer = LogWriter(name=self._env.__class__.__name__)

        check_env(self._env, skip_render_check=True)

    def train(self):
        observation, info = self._env.reset()
        reward = 0
        terminated = truncated = False
        
        step = 0
        while not (terminated or truncated):
            action, values = self._agent.get_action(observation, reward)
            observation, reward, terminated, truncated, info = self._env.step(action)

            self._log_writer.write(values, step)

            step += 1

        self._log_writer.close()