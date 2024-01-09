from colorsys import hsv_to_rgb
from typing import Tuple, Dict, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class LinearSpectrumEnvironment(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=255, shape=(3,), dtype=np.uint8)
        self.STEP_MAX = 5000
        self.STEP_SIZE = 5

        self.state = 128
        self._step = 0

    def reset(
            self, 
            seed=None, 
            options=None
            ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.state = 128
        self._step = 0

        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(
            self, 
            action: np.ndarray
            ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = action.item() 
        # 0: left, 1: right
        assert (action == 0) or (action == 1)

        # take a step
        if action == 0:
            if self.state - self.STEP_SIZE > 0:
                self.state -= self.STEP_SIZE
        elif action == 1:
            if self.state + self.STEP_SIZE < 255:
                self.state += self.STEP_SIZE

        observation = self._get_observation()
        reward = 0

        # Check terminated
        terminated = False

        # Check truncated
        truncated = False
        self._step += 1
        if self._step == self.STEP_MAX:
            truncated = True

        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass

    def _get_observation(self) -> np.ndarray:
        hue = self.state / 255
        observation = np.array(hsv_to_rgb(hue, 1, 1)) * 255
        observation = observation.astype(np.uint8)
        return observation
    
    def _get_info(self) -> Dict[str, Any]:
        info = {
            'Environment.name': 'LinearSpectrumEnvironment',
            'TimeLimit.truncated': self.STEP_MAX,
        }
        return info