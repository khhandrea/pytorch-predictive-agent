from random import choice

import numpy as np

from gymnasium.utils.env_checker import check_env

class Trainer:
    def __init__(self, env):
        check_env(env, skip_render_check=True)
        self._env = env

    def train(self):
        observation, info = self._env.reset()

        for _ in range(500):
            action = np.array(choice((0, 1)))
            observation, reward, terminated, truncated, info = self._env.step(action)
            print(reward, terminated or truncated, self._env.state, observation)