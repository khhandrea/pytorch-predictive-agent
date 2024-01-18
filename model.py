from typing import Tuple

import numpy as np
from torch import nn
from torch import tensor, Tensor, from_numpy

class FeatureExtractorInverseNetwork(nn.Module):
    def __init__(self, 
                 is_linear: bool=False,
                 layer_shape: Tuple=None
                 ):
        super().__init__()
        self._prev_feature = None
        self._feature_extractor = nn.Sequential()
        self._inverse_network = nn.Sequential()

    def forward(self, 
                 observation: Tensor, 
                 prev_action: Tensor,
                 ) -> Tuple[Tensor, float]:
        feature = tensor([])
        inverse_loss = 0.

        return feature, inverse_loss

class PredictorNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, 
                     feature: Tensor, 
                     prev_action: Tensor,
                     ) -> Tuple[Tensor, float, float]:
        z1 = tensor([])
        predictor_loss = 0.
        extrinsic_reward = 0.

        return z1, extrinsic_reward, predictor_loss

class ControllerNetwork(nn.Module):
    def __init__(self, action_space, random_policy):
        super().__init__()
        self._action_space = action_space
        self._random_policy = random_policy

    def forward(self, 
                z: Tensor,
                feature: Tensor,
                extra: Tensor,
                reward: float) -> Tuple[Tensor, float, float]:
        policy_loss = 0.
        value_loss = 0.
        if self._random_policy:
            action = self._action_space.sample()
            action = from_numpy(np.asarray(action))
            
        return action, policy_loss, value_loss