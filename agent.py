from itertools import chain
import os
from typing import Tuple, List, Dict, Any

import numpy as np
import torch
from torch import nn, optim, Tensor
from torch.nn import functional as F
from torch.distributions.categorical import Categorical

from model import ControllerNetwork
from models import MLP, DiscreteLinearActorCritic

class PredictiveAgent:
    def __init__(self, 
                 observation_space,
                 action_space, 
                 random_policy: bool,
                 path: str,
                 device: str,
                 lr_args: Tuple[float, float, float],
                 hidden_state_size: int,
                 feature_size: int,
                 predictor_RNN_num_layers: int,
                 feature_extractor_layerwise_shape: List,
                 inverse_network_layerwise_shape: List):
        if device != 'cpu':
            assert torch.cuda.is_available()
        self._device = torch.device(device)
        print(f"device: {self._device}")

        self._action_space = action_space
        self._path = path
        self._prev_observation = torch.zeros(1, observation_space.shape[0]).to(self._device)
        self._prev_action = torch.zeros(1, self._action_space.n).to(self._device)
        self._inner_state = torch.zeros(1, hidden_state_size).to(self._device)
        feature_extractor_inverse_lr, predictor_lr, controller_lr = lr_args

        self._feature_extractor = MLP(feature_extractor_layerwise_shape, normalize_input=True).to(self._device)
        self._inverse_network = MLP(inverse_network_layerwise_shape, end_with_softmax=True).to(self._device)
        self._feature_predictor = nn.Linear(hidden_state_size + self._action_space.n, feature_size).to(self._device)
        self._inner_state_predictor = nn.LSTM(
            input_size = self._action_space.n + feature_size,
            hidden_size = hidden_state_size,
            num_layers = predictor_RNN_num_layers
        ).to(self._device)
        self._controller = ControllerNetwork(
            self._action_space, 
            random_policy
            ).to(self._device)
        self._controller_agent = ControllerAgent(
            action_space=self._action_space,
            random_policy=random_policy,
            device=self._device,
            gamma=0.99
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
        inverse_loss = self._loss_ce(self._prev_action, pred_prev_action)
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
                   ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # ICM
        observation = torch.from_numpy(observation).float().to(self._device).view(1, -1)
        feature, inverse_loss = self._update_and_get_icm(observation)

        # Predictor
        predictor_loss = self._update_and_get_predictor(feature)
        intrinsic_reward = predictor_loss
        reward = extrinsic_reward + intrinsic_reward

        # Controller
        action, policy_loss, value_loss = self._controller_agent.get_action_and_update(self._inner_state.detach(), reward)
        self._prev_action = F.one_hot(action, num_classes=self._action_space.n).float().to(self._device).view(1, -1)

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

class ControllerAgent:
    def __init__(self,
                 action_space, 
                 random_policy: bool, 
                 device: torch.device,
                 gamma: float
                 ):
        self._action_space = action_space
        self._random_policy = random_policy
        self._device = device
        self._actor_critic = DiscreteLinearActorCritic(
            (128, 128, 64), 
            action_space=action_space).to(self._device)
        self._gamma = gamma

        self._loss_mse = nn.MSELoss()
        self._prev_input = torch.zeros(1, 128).to(self._device)
        self._log_prob = torch.tensor(0).to(self._device)
        self._controller_optimizer = optim.Adam(self._actor_critic.parameters(), lr=0.001)

    def get_action_and_update(self, input: Tensor, reward: float) -> tuple[Tensor, float, float]:
        if self._random_policy:
            policy_loss = 0.
            value_loss = 0.
            random_action = self._action_space.sample()
            action = torch.tensor(random_action, device=self._device)
        else:
            # Update
            self._controller_optimizer.zero_grad()
            policy, value = self._actor_critic(input)
            _, prev_value = self._actor_critic(self._prev_input)
            advantage = reward + self._gamma * value - prev_value
            policy_loss_tensor = -advantage * self._log_prob
            value_loss_tensor = self._loss_mse(reward + self._gamma * value, prev_value)
            loss = policy_loss_tensor + 0.1 * value_loss_tensor
            loss.backward()
            self._controller_optimizer.step()
            self._prev_input = input.detach()

            policy_loss = policy_loss_tensor.item()
            value_loss = value_loss_tensor.item()

            # Get an action
            distribution = Categorical(probs=policy)
            action = distribution.sample()
            self._log_prob = distribution.log_prob(action).detach()
        return action, policy_loss, value_loss