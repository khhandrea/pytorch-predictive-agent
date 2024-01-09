import gymnasium as gym
from gymnasium import spaces
import numpy as np

class LinearSpectrumEnvironment(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=255, shape=(3,), dtype=np.uint8)
        self.STEP_MAX = 5000

        self._state = 128
        self._step = 0

    def reset(self, seed=None, options=None):
        self._state = 128
        self._step = 0

        observation = self._get_observation()
        info = None
        return observation, info

    def step(self, action: np.ndarray):
        action = action.item() 
        # 0: left, 1: right
        assert (action == 0) or (action == 1)

        # take a step
        if action == 0:
            if self._state > 0:
                self._state -= 1
        elif action == 1:
            if self._state < 255:
                self._state += 1

        observation = self._get_observation()
        reward = 0

        # Check terminated
        terminated = False

        # Check truncated
        truncated = False
        self._step += 1
        if self._step == self.STEP_MAX:
            truncated = True

        info = None

        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass

    def _get_observation(self) -> np.ndarray:
        return np.array([self._state] * 3, dtype=np.uint8)