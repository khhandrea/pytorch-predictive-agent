from itertools import chain
import os
from typing import Any

import numpy as np
import torch
from torch import nn, optim, Tensor
from torch.nn import functional as F

from controller_agent import ControllerAgent
from models import MLP, SimpleCNN

class PredictiveAgent:
    def __init__(self, 
                 observation_space,
                 action_space, 
                 random_policy: bool,
                 path: str,
                 device: str,
                 lr_args: tuple[float, float, float],
                 hidden_state_size: int,
                 feature_size: int,
                 predictor_RNN_num_layers: int,
                 feature_extractor_layerwise_shape: tuple,
                 inverse_network_layerwise_shape: tuple,
                 controller_network_layerwise_shape: tuple):
        if device != 'cpu':
            assert torch.cuda.is_available()
        self._device = torch.device(device)
        print(f"device: {self._device}")

        self._action_space = action_space
        self._observation_space = observation_space
        self._path = path
        self._prev_observation = torch.zeros(1, 3, 64, 64).to(self._device)
        self._prev_action = torch.zeros(1, self._action_space.n).to(self._device)
        self._inner_state = torch.zeros(1, hidden_state_size).to(self._device)
        feature_extractor_inverse_lr, predictor_lr, controller_lr = lr_args

        self._feature_extractor = SimpleCNN().to(self._device)
        self._inverse_network = MLP(inverse_network_layerwise_shape, end_with_softmax=True).to(self._device)
        self._feature_predictor = nn.Linear(hidden_state_size + self._action_space.n, feature_size).to(self._device)
        self._inner_state_predictor = nn.LSTM(
            input_size = self._action_space.n + feature_size,
            hidden_size = hidden_state_size,
            num_layers = predictor_RNN_num_layers
        ).to(self._device)
        self._controller_agent = ControllerAgent(
            action_space=self._action_space,
            random_policy=random_policy,
            device=self._device,
            gamma=0.99,
            controller_lr=controller_lr,
            feature_size=feature_size,
            controller_network_layerwise_shape=controller_network_layerwise_shape
        )

        self._loss_ce = nn.CrossEntropyLoss()
        self._loss_mse = nn.MSELoss()

        icm = chain(self._feature_extractor.parameters(), self._inverse_network.parameters())
        predictor = chain(self._feature_predictor.parameters(), self._inner_state_predictor.parameters())
        self._feature_extractor_optimizer = optim.Adam(icm, lr=feature_extractor_inverse_lr)
        self._predictor_optimizer = optim.Adam(predictor, lr=predictor_lr)

    def _update_and_get_icm(self, observation: Tensor) -> tuple[Tensor, float]:
        self._feature_extractor_optimizer.zero_grad()
        prev_feature = self._feature_extractor(self._prev_observation)
        feature = self._feature_extractor(observation)
        concatenated_feature = torch.cat((prev_feature, feature), 1)
        pred_prev_action = self._inverse_network(concatenated_feature)
        inverse_loss = self._loss_ce(pred_prev_action, self._prev_action)
        inverse_loss.backward()
        self._feature_extractor_optimizer.step()
        self._prev_observation = observation
        return feature, inverse_loss.item()

    def _update_and_get_predictor(self, feature: Tensor) -> float:
        self._predictor_optimizer.zero_grad()
        feature = feature.detach()
        inner_state_action = torch.cat((self._inner_state, self._prev_action), 1)
        pred_feature = self._feature_predictor(inner_state_action)
        action_feature = torch.cat((self._prev_action, feature), 1)
        inner_state, _ = self._inner_state_predictor(action_feature)
        predictor_loss = self._loss_mse(feature, pred_feature)
        predictor_loss.backward()
        self._predictor_optimizer.step()
        self._inner_state = inner_state.detach()
        
        return predictor_loss.item()

    def _update_and_get_controller():
        pass

    def get_action(self, 
                   observation: np.ndarray, 
                   extrinsic_reward: float,
                   extra: np.ndarray,
                   ) -> tuple[np.ndarray, dict[str, Any]]:
        # ICM after Normalize
        observation = observation.astype(float) / (self._observation_space.high - self._observation_space.low) + self._observation_space.low
        observation = torch.from_numpy(observation).float().to(self._device).unsqueeze(0)
        feature, inverse_loss = self._update_and_get_icm(observation)

        # Predictor
        predictor_loss = self._update_and_get_predictor(feature)
        intrinsic_reward = predictor_loss
        reward = extrinsic_reward + intrinsic_reward

        # Controller
        action, policy_loss, value_loss = self._controller_agent.get_action_and_update(self._inner_state.detach(), reward)
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

    def load(self, load_args: tuple[str, str, str, str]):
        load, load_inverse, load_predictor, load_controller = load_args

        # Load models from one directory
        if load != 'None':
            self._feature_extractor.load_state_dict(
                torch.load(self._get_load_path(load, 'feature-extractor-inverse-network')))
            self._predictor.load_state_dict(
                torch.load(self._get_load_path(load, 'predictor-network')))
            self._controller.load_state_dict(
                torch.load(self._get_load_path(load, 'controller-network')))
            
        # Load models from each files
        if load_inverse != 'None':
            self._feature_extractor.load_state_dict(
                torch.load(self._get_load_path(load_inverse, 'feature-extractor-inverse-network')))
        if load_predictor != 'None':
            self._predictor.load_state_dict(
                torch.load(self._get_load_path(load_predictor, 'predictor-network')))
        if load_controller != 'None':
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
