from typing import Tuple, Dict, Any

import numpy as np
from torch import nn

class FeatureExtractorNetwork(nn.Module):
    def __init__(self):
        pass

    def feed_obs(self, observation: np.ndarray, prev_action: np.ndarray) -> Tuple[np.ndarray, float]:
        feature = np.array([])
        loss = 0

        return feature, loss

class PredictorNetwork(nn.Module):
    def __init__(self):
        pass

    def feed_feature(self, feature: np.ndarray, prev_action) -> Tuple[np.ndarray, float]:
        z1 = np.array([])
        loss = 0

        return z1, loss

class ControllerNetwork(nn.Module):
    def __init__(self, action_space, random_policy):
        self._action_space = action_space
        self._random_policy = random_policy

    def get_action(self, input_feature: np.ndarray):
        if self._random_policy:
            return self._action_space.sample()
        
    def update(self, reward: float) -> Tuple[float, float]:
        policy_loss = 0
        value_loss = 0

        return policy_loss, value_loss

class PredictAgent:
    def __init__(self, action_space, random_policy):
        self._random_policy = random_policy
        self._action_space = action_space
        
        self._prev_action = None
        self._feature_extractor = FeatureExtractorNetwork()
        self._predictor = PredictorNetwork()
        self._controller = ControllerNetwork(action_space, random_policy)

    def get_action(self, observation: np.ndarray, reward: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        feature, extractor_loss = self._feature_extractor.feed_obs(observation, self._prev_action)
        z1, predictor_loss = self._predictor.feed_feature(feature, self._prev_action)
        policy_loss, value_loss = self._controller.update(reward)
        action = self._controller.get_action(np.concatenate((feature, z1), axis=0))

        values = {
            'loss/extractor_loss': extractor_loss,
            'loss/predictor_loss': predictor_loss,
            'loss/policy_loss': policy_loss,
            'loss/value_loss': value_loss,
        }
        return action, values