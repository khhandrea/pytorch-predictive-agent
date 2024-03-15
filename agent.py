from itertools import chain
from typing import Any

import numpy as np
import torch
from torch import Tensor, optim
from torch.nn import functional as F

from controller_agent import ControllerAgent

class PredictiveAgent:
    def __init__(
        self, 
        device: torch.device,
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
        # self._action_space = action_space
        # self._observation_space = observation_space
        # self._path = path
        # observation_shape = tuple([1] + list(self._observation_space.shape))
        # self._prev_observation = torch.zeros(observation_shape).to(self._device)
        # self._prev_action = torch.zeros(1, self._action_space.n).to(self._device)
        # self._inner_state = torch.zeros(1, hidden_state_size).to(self._device)
        # self.optimizer = torch.zeros(1, hidden_state_size).to(self._device)
        # feature_extractor_module, inverse_module, feature_predictor_module,\
        #     inner_state_predictor_module, controller_module = module_args
        # inverse_module_predictor_lr, controller_lr = lr_args
        # self._intrinsic_reward_discount = intrinsic_reward_discount
        # self._predictor_loss_discount = predictor_loss_discount
        
        if optimizer == 'adam':
            optimizer = optim.Adam
        elif optimizer == 'sgd':
            optimizer = optim.SGD
        else:
            raise Exception(f'Invalid optimizer: {optimizer}')

        # self._feature_extractor = get_class_from_module('models', feature_extractor_module)().to(self._device)
        # self._inverse_network = get_class_from_module('models', inverse_module)().to(self._device)
        # self._feature_predictor = get_class_from_module('models', feature_predictor_module)().to(self._device)
        # self._inner_state_predictor = get_class_from_module('models', inner_state_predictor_module)().to(self._device)
        # self._controller_agent = ControllerAgent(
        #     feature_size=feature_size,
        #     action_space=self._action_space,
        #     random_policy=random_policy,
        #     controller_module=controller_module,
        #     gamma=gamma,
        #     lr=controller_lr,
        #     optimizer_arg=optimizer_args[1],
        #     policy_discount=policy_discount,
        #     entropy_discount=entropy_discount,
        #     device=self._device,
        #     path=self._path,
        #     )

        # self._loss_ce = nn.CrossEntropyLoss()
        # self._loss_mse = nn.MSELoss()

        # models = chain(
        #     self._feature_extractor.parameters(),
        #     self._inverse_network.parameters(),
        #     self._feature_predictor.parameters(),
        #     self._inner_state_predictor.parameters()
        # )
        # self._optimizer = optimizer(models, lr=inverse_module_predictor_lr)

    def get_action(self, 
                   observation: np.ndarray, 
                   extrinsic_reward: float,
                   ) -> tuple[np.ndarray, dict[str, Any], bool]:
        # Preprocess observation
        # Normalize observation
        observation = observation.astype(float) / (self._observation_space.high - self._observation_space.low) + self._observation_space.low
        observation = torch.from_numpy(observation).float().to(self._device).unsqueeze(0)

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

        return action

    def train(
        self,
        batch: Tensor
    ) -> dict[str, Any]:
        data = {}
        data['controller/entropy'] = 0
        data['controller/policy_loss'] = 0
        data['controller/value_loss'] = 0
        data['icm/inverse_accuracy'] = 0
        data['icm/predictor_loss'] = 0

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