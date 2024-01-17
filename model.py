from typing import Tuple, Dict, Any

import numpy as np
from torch import nn

class FeatureExtractorInverseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self._prev_feature = None

    def forward(self, 
                 observation: np.ndarray, 
                 prev_action: np.ndarray,
                 ) -> Tuple[np.ndarray, float]:
        feature = np.array([])
        inverse_loss = 0.

        return feature, inverse_loss

class PredictorNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, 
                     feature: np.ndarray, 
                     prev_action: np.ndarray,
                     ) -> Tuple[np.ndarray, float]:
        z1 = np.array([])
        predictor_loss = 0.
        extrinsic_reward = 0.

        return z1, extrinsic_reward, predictor_loss

class ControllerNetwork(nn.Module):
    def __init__(self, action_space, random_policy):
        super().__init__()
        self._action_space = action_space
        self._random_policy = random_policy

    def forward(self, 
                z: np.ndarray,
                feature: np.ndarray,
                extra: np.ndarray,
                reward: float) -> np.ndarray:
        policy_loss = 0.
        value_loss = 0.
        if self._random_policy:
            action = self._action_space.sample()
            
        return action, policy_loss, value_loss