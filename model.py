from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch import tensor, Tensor

class FeatureExtractorInverseNetwork(nn.Module):
    def __init__(self, 
                 observation_space,
                 action_space,
                 is_linear: bool,
                 feature_extractor_layerwise_shape: Tuple,
                 inverse_network_layerwise_shape: Tuple
                 ):
        """
        Args:
            - observation_space
            - action_space
            - is_linear (bool): 1D np.ndarray or 2D(RGBA) np.ndarray
            - feautre_extractor_layerwise_shape (Tuple): \
                layerwise feature shape of feature extractor. \
                E.g. 1d array: [middle_shape, …, middle_shape, last_shape], \
                2d array: [(channel, kernel, stride), …, (channel, kernel, stride), last_shape]. \
                Warning; don't put first shape!
            - inverse_network_layerwise_shape (Tuple): \
                layerwise feature shape of inverse network. \
                E.g. [middle_shape, …, middle_shape]. \
                Warning; don’t put first/last shape!

        Raises:
            - Invalid 'layerwise_shape' format (todo)
        """
        super().__init__()
        self._prev_observation = None
        self._feature_extractor = nn.Sequential()
        self._inverse_network = nn.Sequential()

        if is_linear:
            feature_extractor_sequence = (observation_space.shape[0],) + feature_extractor_layerwise_shape
            print(feature_extractor_sequence)
            self._feature_extractor.add_module(
                "layer0-linear",
                nn.Linear(feature_extractor_sequence[0], feature_extractor_sequence[1])
            )
            for idx in range(2, len(feature_extractor_sequence)):
                self._feature_extractor.add_module(
                    f"layer{idx - 1}-activation",
                    nn.LeakyReLU()
                )
                self._feature_extractor.add_module(
                    f"layer{idx}-linear", 
                    nn.Linear(feature_extractor_sequence[idx - 1], feature_extractor_sequence[idx])
                )

        inverse_network_sequence = (feature_extractor_sequence[-1] * 2,) + inverse_network_layerwise_shape + (action_space.n,)
        self._inverse_network.add_module(
            "layer0-linear",
            nn.Linear(inverse_network_sequence[0], inverse_network_sequence[1])
        )
        for idx in range(2, len(inverse_network_sequence)):
            self._inverse_network.add_module(
                f"layer{idx - 1}-activation", 
                nn.LeakyReLU()
            )
            self._inverse_network.add_module(
                f"layer{idx}-linear",
                nn.Linear(inverse_network_sequence[idx - 1], inverse_network_sequence[idx])
            )
        self._inverse_network.add_module(
            f"layer{len(inverse_network_sequence) - 1}-activation",
            nn.Softmax(dim=0)
        )

    def forward(self, 
                 observation: Tensor
                 ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            - observation (Tensor)
            - prev_action (Tensor)

        Returns:
            - feature (Tensor)
            - pred_action (Tensor)
        """
        pred_action = None

        if self._prev_observation is None:
            feature = self._feature_extractor(observation)
        else:
            prev_feature = self._feature_extractor(self._prev_observation)
            feature = self._feature_extractor(observation)
            inverse_input = torch.cat((prev_feature.view(-1, 1), feature.view(-1, 1)), 0).view(-1)
            pred_action = self._inverse_network(inverse_input)
        self._prev_observation = observation

        return feature, pred_action

class PredictorNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, 
                     feature: Tensor, 
                     prev_action: Tensor,
                     ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            - feature (Tensor)
            - pred_feature (Tensor)

        Returns:
            - z (Tensor)
            - pred_action (Tensor)
        """
        z = tensor([])
        pred_feature = tensor([])

        return z, pred_feature

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
        """
        Args:
            - action_space
            - random_policy (bool)

        Returns:
            - action (Tensor)
            - policy_loss (float)
            - value_loss (float)
        """
        values = tensor([])
        if self._random_policy:
            action = self._action_space.sample()
            action = torch.from_numpy(np.asarray(action))
            
        return action, values