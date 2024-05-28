from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import deepmind_lab

def _action(*entries):
    return np.array(entries, dtype=np.intc)

# 8 actions
ACTIONS = {
    'look_left': _action(-20, 0, 0, 0, 0, 0, 0),
    'look_right': _action(20, 0, 0, 0, 0, 0, 0),
    'look_up': _action(0, 10, 0, 0, 0, 0, 0),
    'look_down': _action(0, -10, 0, 0, 0, 0, 0),
    'strafe_left': _action(0, 0, -1, 0, 0, 0, 0),
    'strafe_right': _action(0, 0, 1, 0, 0, 0, 0),
    'forward': _action(0, 0, 0, 1, 0, 0, 0),
    'backward': _action(0, 0, 0, -1, 0, 0, 0),
    # 'fire': _action(0, 0, 0, 0, 1, 0, 0),
    # 'jump': _action(0, 0, 0, 0, 0, 1, 0),
    # 'crouch': _action(0, 0, 0, 0, 0, 0, 1)
}

class DeepmindLabEnvironment(gym.Env):
    metadata = {
        'render_modes': ['none'],
        'render_fps': 30
    }

    def __init__(self, 
                 level: str,
                 step_max: int,
                 observation_type: str,
                 **config: dict[str, Any]):
        self._level = level
        self._step_max = step_max
        self._total_step = 0
        self._observation_type = observation_type
        self._env = deepmind_lab.Lab(self._level,
                                     [observation_type,
                                      'DEBUG.POS.TRANS', 'DEBUG.POS.ROT',
                                      'VEL.TRANS', 'VEL.ROT'],
                                     config=config)

        self._action_list = list(ACTIONS.values())
        self.action_space = spaces.Discrete(len(self._action_list))
        if self._observation_type == 'RGB':
            self.observation_space = spaces.Box(low=0, high=255, shape=(3, int(config['width']), int(config['height'])), dtype=np.uint8)

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict[str, Any]]:
        self._env.reset()
        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(self,
             action: int
             ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        reward = self._env.step(self._action_list[action], num_steps=1)
        if not self._env.is_running():
            self._env.reset()
        truncated = (self._total_step > self._step_max)
        terminated = False
        observation = self._get_observation()
        info = self._get_info()
        self._total_step += 1
        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass

    def _get_observation(self) -> np.ndarray:
        return self._env.observations()[self._observation_type]

    def _get_info(self) -> dict[str, Any]:
        info = {
            'level': self._level,
            'position': self._env.observations()['DEBUG.POS.TRANS'],
            'rotation': self._env.observations()['DEBUG.POS.ROT'],
            'velocity': self._env.observations()['VEL.TRANS'],
            'angular_velocity': self._env.observations()['VEL.ROT'],
            'Environment.coordinate': self._env.observations()['DEBUG.POS.TRANS'][0:1]
            }
        return info