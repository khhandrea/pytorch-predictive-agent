import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np

class VisualCartPole(gym.Env):
    metadata = {
    'render_modes': ['none', 'human', 'rgb_array'],
        'render_fps': 30,
    }

    def __init__(self, step_max: int):
        self._env = gym.make('CartPole-v1', render_mode='rgb_array')
        
        self.action_space = self._env.action_space
        self.observation_space = Box(0, 255, shape=(3, 600, 400), dtype=np.uint8)

    def reset(self):
        _, info = self._env.reset()
        observation = self.render()
        return observation, info

    def step(self, action):
        _, reward, truncated, terminated, info = self._env.step(action)
        observation = self.render()
        return observation, reward, truncated, terminated, info
    
    def render(self):
        output = self._env.render()
        output = output.transpose(2, 1, 0)
        return output

    def close(self):
        output = self._env.close()
        return output