import os
from typing import Tuple, Dict, Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

from model import FeatureExtractorInverseNetwork, PredictorNetwork, ControllerNetwork

class PredictiveAgent:
    def __init__(self, 
                 observation_space,
                 action_space, 
                 random_policy: bool,
                 path: str):
        self._prev_action = None
        self._action_space = action_space
        self._path = path

        self._feature_extractor = FeatureExtractorInverseNetwork(
            observation_space=observation_space,
            action_space=self._action_space,
            is_linear=True,
            feature_extractor_layerwise_shape=(64, 128),
            inverse_network_layerwise_shape=(128,))
        self._predictor = PredictorNetwork()
        self._controller = ControllerNetwork(self._action_space, random_policy)

        self._feature_extractor_optimizer = optim.Adam(self._feature_extractor.parameters(), lr=1e-4)

    def get_action(self, 
                   observation: np.ndarray, 
                   extrinsic_reward: float,
                   extra: np.ndarray,
                   ) -> Tuple[np.ndarray, Dict[str, Any]]:
        observation = torch.from_numpy(observation).float()
        loss_ce = nn.CrossEntropyLoss()
        loss_mse = nn.MSELoss()

        # Feed and update the feature extractor inverse network
        self._feature_extractor_optimizer.zero_grad()
        feature, pred_prev_action = self._feature_extractor(observation)
        if self._prev_action is None:
            inverse_loss = 0.
        else:
            inverse_loss = loss_ce(self._prev_action, pred_prev_action)
            inverse_loss.backward()
            self._feature_extractor_optimizer.step()

        # Feed and update the predictor network
        z, pred_feature = self._predictor(feature, self._prev_action)
        # predictor_loss = loss_mse(feature, pred_feature)
        predictor_loss = 0.
        intrinsic_reward = predictor_loss
        reward = extrinsic_reward + intrinsic_reward

        # Get a action and update the controller network
        policy_loss = 0.
        value_loss = 0.
        action, values = self._controller(z, feature, extra, reward)
        self._prev_action = F.one_hot(action, num_classes=self._action_space.n).float()

        values = {
            'loss/inverse_loss': inverse_loss,
            'loss/predictor_loss': predictor_loss,
            'loss/policy_loss': policy_loss,
            'loss/value_loss': value_loss,
            'reward/intrinsic_reward': intrinsic_reward
        }

        return action, values

    def _get_load_path(self, load_arg: str, network: str) -> str:
        environment, description, step = load_arg.split('/')
        return os.path.join('checkpoints', environment, description, network, f'step-{step}') + '.pt'

    def load(self, load_args: Tuple[str, str, str, str]):
        load, load_inverse, load_predictor, load_controller = load_args
        if load is None:
            if (load_inverse is not None) \
                and (load_predictor is not None) \
                and (load_controller is not None):
                # Load models from each files
                self._feature_extractor.load_state_dict(
                    torch.load(self._get_load_path(load_inverse, 'feature-extractor-inverse-network')))
                self._predictor.load_state_dict(
                    torch.load(self._get_load_path(load_predictor, 'predictor-network')))
                self._controller.load_state_dict(
                    torch.load(self._get_load_path(load_controller, 'controller-network')))
            elif (load_inverse, load_predictor, load_controller) == (None, None, None):
                return
            else:
                raise Exception("Any of '--load_inverse', '--load_predictor' "
                                + " or '--load_controller' options are missing")
        else:
            self._feature_extractor.load_state_dict(
                torch.load(self._get_load_path(load, 'feature-extractor-inverse-network')))
            self._predictor.load_state_dict(
                torch.load(self._get_load_path(load, 'predictor-network')))
            self._controller.load_state_dict(
                torch.load(self._get_load_path(load, 'controller-network')))
    
    def _makedir_and_save_model(self, 
                                state_dict, 
                                network: str, 
                                description: str):
        path = os.path.join('checkpoints', self._path, network)
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(
            state_dict,
            os.path.join(path, description) + '.pt'
        )

    def save(self, description: str):
        self._makedir_and_save_model(
            self._feature_extractor.state_dict(),
            'feature-extractor-inverse-network',
            description)
        self._makedir_and_save_model(
            self._predictor.state_dict(),
            'predictor-network',
            description)
        self._makedir_and_save_model(
            self._controller.state_dict(),
            'controller-network',
            description)