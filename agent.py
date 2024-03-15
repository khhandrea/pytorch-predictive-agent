from itertools import chain
from typing import Any

import numpy as np
import torch
from torch import nn, Tensor, tensor, optim
from torch.distributions.categorical import Categorical
from torch.nn import functional as F

from utils import initialize_custom_model
from utils import SharedActorCritic

class PredictiveAgent:
    def __init__(
        self, 
        env,
        device: torch.device,
        network_spec: dict[str, Any],
        global_networks: dict[str, nn.Module],
        random_policy: bool,
        learning_rate: float,
        optimizer: str,
        inverse_loss_scale: float,
        predictor_loss_scale: float,
        value_loss_scale: float,
        policy_loss_scale: float,
        entropy_scale: float,
        gamma: float,
        lmbda: float,
        intrinsic_reward_scale: float,
    ):
        self._random_policy = random_policy
        self._action_space = env.action_space

        if optimizer == 'adam':
            optimizer = optim.Adam
        elif optimizer == 'sgd':
            optimizer = optim.SGD
        else:
            raise Exception(f'Invalid optimizer: {optimizer}')

        # Hyperparameters
        self._inverse_loss_scale = inverse_loss_scale
        self._predictor_loss_scale = predictor_loss_scale
        self._value_loss_scale = value_loss_scale
        self._policy_loss_scale = policy_loss_scale
        self._entropy_scale = entropy_scale
        self._gamma = gamma
        self._lmbda = lmbda
        self._intrinsic_reward_scale = intrinsic_reward_scale

        # Initialize global and local networks
        networks = (
            'feature_extractor',
            'inverse_network',
            'inner_state_predictor',
            'feature_predictor',
            'controller'
        )
        self._global_networks = global_networks
        local_networks = {}
        for network in networks[:-1]:
            local_networks[network] = initialize_custom_model(network_spec[network]).to(device)
        local_networks['controller'] = SharedActorCritic(
            shared_network=initialize_custom_model(network_spec['controller_shared']),
            actor_network=initialize_custom_model(network_spec['controller_actor']),
            critic_network=initialize_custom_model(network_spec['controller_critic'])
        ).to(device)

        # Loss functions and an optimizer
        self._loss_ce = nn.CrossEntropyLoss()
        self._loss_mse = nn.MSELoss()
        models = chain(self._global_networks[network].parameters for network in networks)
        self._optimizer = optimizer(models, lr=learning_rate)

        # Initialize prev data
        self._prev_action = tensor.zeros(1, self._action_space.n).to(device)
        _, self._prev_inner_state = self._local_networks['inner_state_predictor'](torch.zeros(1, 256))

        # Initialize batch_prev data
        self._batch_prev_observation = torch.zeros((1, 3, 64, 64))
        self._batch_prev_actions = torch.zeros(2)
        _, self._batch_prev_inner_state = self._local_networks['inner_state_predictor'](torch.zeros(1, 256))

    def get_action(
        self, 
        observation: np.ndarray, 
    ) -> np.ndarray:
        if self._random_policy:
            action = np.array(self._action_space.sample())
        else:
            # Preprocess and normalize observation
            observation = observation.astype(float) / (self._observation_space.high - self._observation_space.low) + self._observation_space.low
            observation = torch.from_numpy(observation).float().to(self._device).unsqueeze(0)

            # Inference with inverse module
            feature = self._local_networks['feature_extractor'](observation)

            # Inference with inner state predictor
            action_feature = torch.cat((self._prev_action, feature.detach()), 1)
            inner_state, self._prev_inner_state = self._inner_state_predictor(action_feature, self._prev_inner_state)

            # Controller
            policy, _ = self._local_networks['controller'](inner_state)
            distribution = Categorical(probs=policy)
            action = distribution.sample()
            
            # Update
            self._prev_action = F.one_hot(action.detach(), num_classes=self._action_space.n).float() # .to(self._device)

        return action

    def train(
        self,
        batch: dict[str, Tensor]
    ) -> dict[str, float]:
        data = {}
        data['controller/entropy'] = 0.
        data['controller/policy_loss'] = 0.
        data['controller/value_loss'] = 0.
        data['icm/inverse_accuracy'] = 0.
        data['icm/predictor_loss'] = 0.

        return data
        
        # Initialize optimizers
        self._optimizer.zero_grad()

        # Inference with inverse module
        prev_feature = self._feature_extractor(self._prev_observation)
        feature = self._feature_extractor(observation)
        concatenated_feature = torch.cat((prev_feature, feature), 1)
        pred_prev_action = self._inverse_network(concatenated_feature)

        self._prev_observation = observation

        # Inference with feature predictor
        inner_state_action = torch.cat((self._inner_state, self._prev_action), 1)
        pred_feature = self._feature_predictor(inner_state_action)


        # Inference with inner state predictor
        action_feature = torch.cat((self._prev_action, feature.detach()), 1)
        self._inner_state = self._inner_state_predictor(action_feature)

        # Controller
        action, policy_loss_item, value_loss_item, entropy_item = self._controller_agent.get_action_and_update(self._inner_state.detach(), reward)
        self._prev_action = F.one_hot(action.detach(), num_classes=self._action_space.n).float().to(self._device)

        # Update modules
        inverse_loss = self._loss_ce(pred_prev_action, self._prev_action)
        predictor_loss = self._loss_mse(feature.detach(), pred_feature)
        total_loss = inverse_loss + self._predictor_loss_discount * predictor_loss
        total_loss.backward()
        self._optimizer.step()

        # Update values
        correct = torch.argmax(pred_prev_action).item() == torch.argmax(self._prev_action).item()
        inverse_loss_item = inverse_loss.item()
        predictor_loss_item = predictor_loss.item()
        intrinsic_reward = self._intrinsic_reward_discount * predictor_loss_item
        reward = extrinsic_reward + intrinsic_reward