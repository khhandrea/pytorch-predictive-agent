from typing import Tuple, Dict, Any

import numpy as np
import torch

from model import FeatureExtractorInverseNetwork, PredictorNetwork, ControllerNetwork

class PredictiveAgent:
    def __init__(self, action_space, random_policy):
        self._prev_action = None
        self._feature_extractor = FeatureExtractorInverseNetwork(
            action_shape=action_space,
            is_linear=True,
            layerwise_shape=(3, 3))
        self._predictor = PredictorNetwork()
        self._controller = ControllerNetwork(action_space, random_policy)

    def get_action(self, 
                   observation: np.ndarray, 
                   extrinsic_reward: float,
                   extra: np.ndarray,
                   ) -> Tuple[np.ndarray, Dict[str, Any]]:
        observation = torch.from_numpy(observation).float()
        feature, inverse_loss = self._feature_extractor.feed(observation, self._prev_action)
        z, intrinsic_reward, predictor_loss = self._predictor.feed(feature, self._prev_action)
        reward = extrinsic_reward + intrinsic_reward
        action, policy_loss, value_loss = self._controller.feed(z, feature, extra, reward)
        self._prev_action = action

        values = {
            'loss/inverse_loss': inverse_loss,
            'loss/predictor_loss': predictor_loss,
            'loss/policy_loss': policy_loss,
            'loss/value_loss': value_loss,
            'reward/intrinsic_reward': intrinsic_reward
        }

        return action, values