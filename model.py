from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch import tensor, Tensor
from torch import optim

class FeatureExtractorInverseNetwork(nn.Module):
    def __init__(self, 
                 action_shape,
                 is_linear: bool=False,
                 layerwise_shape: Tuple=None
                 ):
        """
        Args:
            - is_linear (bool): 1D np.ndarray or 2D(RGBA) np.ndarray
            - layerwise_shape (Tuple): layerwise feature shape.\
                E.g. 1d array: [first_shape, middle_shape, …, middle_shape, last_shape],\
                2d array: [first_channel, (channel, kernel, stride), …, (channel, kernel, stride), last_shape]

        Raises:
            - Expected 'layerwise_shape' argument if is_linear is True
            - Invalid 'layerwise_shape' format (todo)
        """
        if is_linear and layerwise_shape is None:
            raise Exception("Expected 'layerwise_shape' argument if is_linear is True")
        
        super().__init__()
        self._prev_observation = None
        self._inverse_network = nn.Linear(layerwise_shape[-1] * 2, action_shape.n)
        self._feature_extractor = nn.Sequential()

        if is_linear:
            for idx in range(1, len(layerwise_shape)):
                self._feature_extractor.add_module(
                    f"layer{idx}", nn.Linear(layerwise_shape[idx - 1], layerwise_shape[idx])
                )

        self._optimizer = optim.Adam(
            list(self._inverse_network.parameters())
            + list(self._feature_extractor.parameters())
        )
        


    def feed(self, 
                 observation: Tensor, 
                 prev_action: Tensor,
                 ) -> Tuple[Tensor, float]:
        """
        Args:
            - observation (Tensor)
            - prev_action (Tensor)

        Returns:
            - feature (Tensor)
            - inverse_loss (float)
        """
        loss_func = nn.CrossEntropyLoss()
        inverse_loss = 0.

        if self._prev_observation is not None:
            self._optimizer.zero_grad()
            prev_feature = self._feature_extractor(self._prev_observation)
            feature = self._feature_extractor(observation)
            pred_action = self._inverse_network(torch.cat((prev_feature.view(-1, 1), feature.view(-1, 1)), 0).view(-1))
            inverse_loss = loss_func(pred_action, prev_action)
            inverse_loss.backward()
            self._optimizer.step()
        else:
            feature = self._feature_extractor(observation)
        self._prev_observation = observation

        return feature, inverse_loss

class PredictorNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def feed(self, 
                     feature: Tensor, 
                     prev_action: Tensor,
                     ) -> Tuple[Tensor, float, float]:
        """
        Args:
            - feature (Tensor)
            - prev_action (Tensor)

        Returns:
            - z (Tensor)
            - extrinsic_reward (float)
            - predictor_loss (float)
        """
        z = tensor([])
        predictor_loss = 0.
        extrinsic_reward = 0.

        return z, extrinsic_reward, predictor_loss

class ControllerNetwork(nn.Module):
    def __init__(self, action_space, random_policy):
        super().__init__()
        self._action_space = action_space
        self._random_policy = random_policy

    def feed(self, 
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
        policy_loss = 0.
        value_loss = 0.
        if self._random_policy:
            action = self._action_space.sample()
            action = torch.from_numpy(np.asarray(action))
            
        return action, policy_loss, value_loss