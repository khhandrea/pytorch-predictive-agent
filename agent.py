from itertools import chain

import numpy as np
import torch
from torch import nn, Tensor, tensor, optim
from torch.distributions.categorical import Categorical
from torch.nn import functional as F

from utils import CustomModule, SharedActorCritic

def initialize_optimizer(
    optimizer_name: str,
    params,
    learning_rate: float
) -> torch.optim:
    match optimizer_name:
        case 'sgd':
            optimizer = optim.SGD
        case 'adam':
            optimizer = optim.Adam
        case _:
            raise Exception(f'Invalid optimizer: {optimizer}')
    return optimizer(params, lr=learning_rate)


class PredictiveAgent:
    def __init__(
        self, 
        env,
        device: torch.device,
        network_spec: dict[str, dict],
        global_networks: dict[str, nn.Module],
        hyperparameters: dict[str, float],
    ):
        self._device = device
        self._action_space = env.action_space

        # Initialize local networks
        self._networks = (
            'feature_extractor',
            'inverse_network',
            'inner_state_predictor',
            'feature_predictor',
            'controller'
        )
        self._global_networks = global_networks
        self._local_networks = {}
        for network in self._networks[:-1]:
            self._local_networks[network] = CustomModule(network_spec[network])
            self._local_networks[network] = self._local_networks[network].to(self._device)
        self._local_networks['controller'] = SharedActorCritic(
            shared_network=CustomModule(network_spec['controller_shared']),
            actor_network=CustomModule(network_spec['controller_actor']),
            critic_network=CustomModule(network_spec['controller_critic'])
        ).to(self._device)

        # Hyperparameters
        models = chain(*[self._global_networks[network].parameters() for network in self._networks])
        self._optimizer = initialize_optimizer(
            hyperparameters['optimizer'],
            models,
            hyperparameters['learning_rate']
        )
        self._random_policy = hyperparameters['random_policy']
        self._inverse_loss_scale = hyperparameters['inverse_loss_scale']
        self._predictor_loss_scale = hyperparameters['predictor_loss_scale']
        self._value_loss_scale = hyperparameters['value_loss_scale']
        self._policy_loss_scale = hyperparameters['policy_loss_scale']
        self._entropy_scale = hyperparameters['entropy_scale']
        self._gamma = hyperparameters['gamma']
        self._lmbda = hyperparameters['lmbda']
        self._intrinsic_reward_scale = hyperparameters['intrinsic_reward_scale']

        # Loss functions
        self._loss_ce = nn.CrossEntropyLoss()
        self._loss_mse = nn.MSELoss()

        # Initialize prev data
        self._prev_action = torch.zeros(1, self._action_space.n).to(self._device)
        self._prev_inner_state = self._local_networks['inner_state_predictor'](torch.zeros(1, 260))

        # Initialize batch_prev data
        self._batch_prev_observation = torch.zeros((1, 3, 64, 64))
        self._batch_prev_actions = torch.zeros(2)
        self._batch_prev_inner_state = self._local_networks['inner_state_predictor'](torch.zeros(1, 260))

    def get_action(
        self, 
        observation: np.ndarray, 
    ) -> np.ndarray:
        if self._random_policy:
            action = np.array(self._action_space.sample())
        else:
            # Preprocess and normalize observation
            observation = observation.astype(float) / (self._observation_space.high - self._observation_space.low) + self._observation_space.low
            observation = torch.from_numpy(observation).float().unsqueeze(0)

            # Feature extractor
            feature = self._local_networks['feature_extractor'](observation)
            # Inner state predictor
            action_feature = torch.cat((self._prev_action, feature.detach()), 1)
            inner_state, self._prev_inner_state = self._inner_state_predictor(action_feature, self._prev_inner_state)
            # Controller
            policy, _ = self._local_networks['controller'](inner_state)
            distribution = Categorical(probs=policy)
            action = distribution.sample()
            
            self._prev_action = F.one_hot(action.detach(), num_classes=self._action_space.n).float().to(self._device)
        return action

    def sync_network(self) -> None:
        for network in self._networks:
            self._local_networks[network].load_state_dict(self._global_networks[network].state_dict())

    def train(
        self,
        batch: dict[str, Tensor]
    ) -> dict[str, float]:
        # Initialize optimizers
        self._optimizer.zero_grad()

        # Preprocessing
        observations = batch['observations']
        observations = observations / 255

        # From t-1 to T
        observations = torch.cat((self._batch_prev_observation, observations), dim=0)
        # From t-2 to T
        actions = torch.cat((self._batch_prev_actions, batch['actions']), dim=0)
        actions = F.one_hot(actions.to(torch.int64), num_classes=self._action_space.n).float()

        # Inverse loss
        features = self._local_networks['feature_extractor'](observations)
        concatenated_features = torch.cat((features[:-1], features[1:]), dim=1)
        pred_actions = self._local_networks['inverse_network'](concatenated_features)
        inverse_loss = self._loss_ce(actions[1:-1], pred_actions)
        print(torch.argmax(pred_actions, dim=1).tolist())

        # Predictor loss
        detached_features = features.detach()
        detached_actions = actions.detach()
        inner_states = self._local_networks['inner_state_predictor'](
            torch.cat((detached_features, detached_actions[:-1]), dim=1),
            self._batch_prev_inner_state
        )
        self._batch_prev_inner_state = inner_states[-2].unsqueeze(dim=0).detach()
        inner_states_actions = torch.cat((inner_states[:-1], detached_actions[1:-1]), dim=1)
        pred_features = self._local_networks['feature_predictor'](inner_states_actions)
        predictor_loss = self._loss_mse(detached_features[1:], pred_features) * 0

        # Controller loss
        if self._random_policy:
            policy_loss = tensor(0.)
            value_loss = tensor(0.)
            entropy = tensor(0.)
        else:
            # reward = extrinsic_reward + self._intrinsic_reward_scale * intrinsic_reward
            policies, v_preds = self._local_networks['controller'](inner_states[1:].detach())
            distributions = Categorical(probs=policies)
            log_probs = distributions.log_prob(actions[2:].detach())
            entropy = distributions.entropy().mean()
            policy_loss = -(gaes * log_probs).mean()
            value_loss = self._loss_mse(v_targets, v_preds)
        controller_loss = (
            self._policy_loss_scale * policy_loss
            + self._value_loss_scale * value_loss
            + self._entropy_scale * entropy
        )

        # Total loss
        loss = (
            self._inverse_loss_scale * inverse_loss
            + self._predictor_loss_scale * predictor_loss
            + controller_loss
        )
        loss.backward()
        for network in self._networks:
            for global_param, local_param in zip(self._global_networks[network].parameters(), self._local_networks[network].parameters()):
                global_param._grad = local_param.grad
        self._optimizer.step()
        self.sync_network()
        
        # Update memory
        self._batch_prev_observation = observations[-1].detach().unsqueeze(dim=0)
        self._batch_prev_actions = torch.argmax(actions[-2:], dim=1).detach()

        # Process and return data
        inverse_is_correct = torch.argmax(pred_actions, dim=1) == torch.argmax(actions[1:-1], dim=1)
        inverse_accuracy = inverse_is_correct.sum() / len(inverse_is_correct)

        data = {}
        data['controller/entropy'] = entropy.item()
        data['controller/policy_loss'] = policy_loss.item()
        data['controller/value_loss'] = value_loss.item()
        data['icm/inverse_accuracy'] = inverse_accuracy.item()
        data['icm/inverse_loss'] = inverse_loss.item()
        data['icm/predictor_loss'] = predictor_loss.item()

        return data