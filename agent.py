from typing import Tuple, Dict, Any

import numpy as np

class FeatureExtractorNetwork:
    def __init__(self):
        pass

class PredictorNetwork:
    def __init__(self):
        pass

class ControllerNetwork:
    def __init__(self, action_space, random_policy):
        self._action_space = action_space
        self._random_policy = random_policy

    def get_action(self, observation: np.ndarray):
        if self._random_policy:
            return self._action_space.sample()

class PredictAgent:
    def __init__(self, action_space, random_policy):
        self._random_policy = random_policy
        self._action_space = action_space

        self._feature_extractor = FeatureExtractorNetwork()
        self._predictor = PredictorNetwork()
        self._controller = ControllerNetwork(action_space, random_policy)

    def get_action(self, observation: np.ndarray, reward: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        action = self._controller.get_action(observation)
        values = {}
        return action, values