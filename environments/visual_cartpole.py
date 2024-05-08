import gymnasium as gym
from gymnasium.wrappers import PixelObservationWrapper
import numpy as np

class VisualCartpole(gym.Env):
    metadata = {
        'render_modes': ['none', 'human'],
        'render_fps': 30,
    }

    def __init__(self):
        env = gym.make('CartPole-v1', render_mode='rgb_array')
        self._env = PixelObservationWrapper(env)
        
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    def reset(self):
        observation, info = self._env.reset()
        observation = np.transpose(observation['pixels'], (2, 0, 1))
        return observation, info

    def step(self, action):
        observation, reward, truncated, terminated, info = self._env.step(action)
        observation = np.transpose(observation['pixels'], (2, 0, 1))
        return observation, reward, truncated, terminated, info
    
    def render(self):
        output = self._env.render()
        return output

    def close(self):
        output = self._env.close()
        return output