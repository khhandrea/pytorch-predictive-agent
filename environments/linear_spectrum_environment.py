from colorsys import hsv_to_rgb
from typing import Tuple, Dict, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class LinearSpectrumEnvironment(gym.Env):
    metadata = {
        'render_modes': ['human'],
        'render_fps': 15,
        }
    
    def __init__(self, render_mode=None):
        self.STEP_MAX = 5000
        self.STEP_SIZE = 5

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=255, shape=(3,), dtype=np.uint8)

        self.state = 180
        self._step = 0

        # For rendering
        self._render_mode = render_mode
        self._window = None
        self._clock = None

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

        if self._render_mode == 'human':
            self._render_frame()

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

        if self._render_mode == 'human':
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def _draw_rainbow_rect(self, canvas, rect_x, rect_y, rect_width, rect_height):
        for x in range(rect_width):
            hue = x / rect_width
            rgb_color = [int(c * 255) for c in hsv_to_rgb(hue, 1, 1)]

            pygame.draw.rect(canvas, rgb_color, (rect_x + x, 
                                                rect_y, 
                                                1, 
                                                rect_height))

    def _render_frame(self):
        CANVAS_WIDTH = 512
        CANVAS_HEIGHT = 512
        RECT_WIDTH = 360
        RECT_HEIGHT = 64
        rect_x = (CANVAS_WIDTH - RECT_WIDTH) // 2
        rect_y = (CANVAS_HEIGHT - RECT_HEIGHT) // 2
        BORDER_WIDTH = 2
        BORDER_COLOR = (0, 0, 0)
        AGENT_SIZE = 30
        agent_x = rect_x + self.state
        agent_y = rect_y + RECT_HEIGHT + 48

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
        
        canvas.fill((255, 255, 255))

        # Border
        pygame.draw.rect(canvas, BORDER_COLOR, (rect_x - BORDER_WIDTH,
                                                rect_y - BORDER_WIDTH,
                                                RECT_WIDTH + 2 * BORDER_WIDTH,
                                                RECT_HEIGHT + 2 * BORDER_WIDTH))
        
        # Linear spectrum
        self._draw_rainbow_rect(canvas, rect_x, rect_y, RECT_WIDTH, RECT_HEIGHT)

        # Agent
        pygame.draw.polygon(canvas, (0, 0, 255), [(agent_x, agent_y),
                                                (agent_x + AGENT_SIZE, agent_y),
                                                (agent_x + AGENT_SIZE // 2, agent_y - AGENT_SIZE)])

        if self._render_mode == 'human':
            self._window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self._clock.tick(self.metadata['render_fps'])

    def close(self):
        if self._window is not None:
            pygame.display.quit()
            pygame.quit()

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