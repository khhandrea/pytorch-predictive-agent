from random import randint
from typing import Any

import gymnasium as gym
from gymnasium import spaces
from PIL import Image
import numpy as np
import pygame

class MovingImageEnvironment(gym.Env):
    metadata = {
        'render_modes': ['none', 'human'],
        'render_fps': 30
    }

    def __init__(self,
                 render_mode: str,
                 agent_speed: int,
                 step_max: int,
                 noise_scale: float):
        self._image = Image.open('sources/animals.png')
        self.coordinate = [0, 0] # x, y
        self._observation_size = (3, 64, 64)
        self._agent_speed = agent_speed
        self._step = 0
        self._step_max = step_max
        self._noise_scale = noise_scale

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8)

        self._render_mode = render_mode
        self._window = None
        self._clock = None

    def reset(self,
              seed=None,
              options=None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        self.coordinate = [randint(0, self._image.width), randint(0, self._image.height)]
        self._step = 0

        observation = self._get_observation()
        info = self._get_info()

        if self._render_frame == 'human':
            self._render_frame()

        return observation, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = action.item()
        # 0: up, 1: right, 2: down, 3: left
        assert action in [0, 1, 2, 3]

        # Take a step
        # Move up
        if action == 0:
            self.coordinate[1] -= self._agent_speed
        # Move right
        if action == 1:
            self.coordinate[0] += self._agent_speed
        # Move down
        if action == 2:
            self.coordinate[1] += self._agent_speed
        # Move left
        if action == 3:
            self.coordinate[0] -= self._agent_speed

        self.coordinate[0] = max(0, min(self._image.height - self._observation_size[1], self.coordinate[0]))
        self.coordinate[1] = max(0, min(self._image.height - self._observation_size[2], self.coordinate[1]))

        observation = self._get_observation()
        reward = 0
        terminated =False

        # Check truncated
        truncated = False
        self._step += 1
        if self._step == self._step_max:
            truncated = True

        info = self._get_info()

        if self._render_mode == 'human':
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def _render_frame(self):
        CANVAS_WIDTH = 128
        CANVAS_HEIGHT = 128

        if self._render_mode == 'human':
            if self._window is None:
                pygame.init()
                pygame.display.init()
                self._window = pygame.display.set_mode(
                    (CANVAS_WIDTH, CANVAS_HEIGHT)
                )
            
            if self._clock is None:
                self._clock = pygame.time.Clock()

        canvas = pygame.Surface((CANVAS_WIDTH, CANVAS_HEIGHT))

        canvas.fill((0, 0, 0))

        observation = self._get_observation()
        image = np.transpose(observation, (1, 2, 0))
        pygame_surface = pygame.surfarray.make_surface(image)

        if self._render_mode == 'human':
            self._window.blit(pygame_surface, (0, 0))
            pygame.event.pump()
            pygame.display.update()

            self._clock.tick(self.metadata['render_fps'])

        # pygame.display.flip()

    def close(self):
        if self._window is not None:
            pygame.display.quit()
            pygame.quit()

    def _get_observation(self, add_noise: bool = True) -> np.ndarray:
        cropped_image = self._image.crop((
            self.coordinate[0],
            self.coordinate[1],
            self.coordinate[0] + self._observation_size[1],
            self.coordinate[1] + self._observation_size[2]
        ))

        rgb_image = cropped_image.convert('RGB')
        numpy_image = np.array(rgb_image)

        # Transpose the array to have shape (3, width, height)
        numpy_image = np.transpose(numpy_image, (2, 0, 1))

        if add_noise:
            noise = np.random.normal(loc=0, scale=self._noise_scale, size=numpy_image.shape)
            noisy_image = np.clip(numpy_image + noise, 0, 255).astype(np.uint8)
            return noisy_image
        else:
            return numpy_image

    def _get_info(self) -> dict[str, Any]:
        info = {
            'Environment.name': 'LinearSpectrumEnvironment',
            'TimeLimit.truncated': self._step_max
        }
        return info