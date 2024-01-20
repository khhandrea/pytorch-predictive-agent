import os
from typing import Tuple, Dict, Any

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

from model import FeatureExtractorInverseNetwork, PredictorNetwork, ControllerNetwork

class PredictiveAgent:
    def __init__(self, 
                 observation_space,
                 action_space, 
                 random_policy: bool,
                 path: str,
                 cpu: bool):
        if not cpu and torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')
        print(f"device: {self._device}")
        self._action_space = action_space
        self._path = path
        self._prev_action = None

        self._feature_extractor = FeatureExtractorInverseNetwork(
            observation_space=observation_space,
            action_space=self._action_space,
            is_linear=True,
            feature_extractor_layerwise_shape=(64, 128),
            inverse_network_layerwise_shape=(128,)).to(self._device)
        self._predictor = PredictorNetwork().to(self._device)
        self._controller = ControllerNetwork(
            self._action_space, 
            random_policy
            ).to(self._device)

        self._feature_extractor_optimizer = optim.Adam(self._feature_extractor.parameters(), lr=1e-5)

    def get_action(self, 
                   observation: np.ndarray, 
                   extrinsic_reward: float,
                   extra: np.ndarray,
                   ) -> Tuple[np.ndarray, Dict[str, Any]]:
        observation = torch.from_numpy(observation).float().to(self._device)
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
        self._prev_action = F.one_hot(action, num_classes=self._action_space.n).float().to(self._device)

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

        # Load models from one directory
        if load is not None:
            self._feature_extractor.load_state_dict(
                torch.load(self._get_load_path(load, 'feature-extractor-inverse-network')))
            self._predictor.load_state_dict(
                torch.load(self._get_load_path(load, 'predictor-network')))
            self._controller.load_state_dict(
                torch.load(self._get_load_path(load, 'controller-network')))
            
        # Load models from each files
        if load_inverse is not None:
            self._feature_extractor.load_state_dict(
                torch.load(self._get_load_path(load_inverse, 'feature-extractor-inverse-network')))
        if load_predictor is not None:
            self._predictor.load_state_dict(
                torch.load(self._get_load_path(load_predictor, 'predictor-network')))
        if load_controller is not None:
            self._controller.load_state_dict(
                torch.load(self._get_load_path(load_controller, 'controller-network')))
    
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