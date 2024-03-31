from itertools import chain

import numpy as np
import torch
from torch import nn, Tensor, tensor
from torch.distributions.categorical import Categorical
from torch.nn import functional as F

from utils import CustomModule, SharedActorCritic
from utils import initialize_optimizer

class PredictiveAgent:
    def __init__(self, 
                 env,
                 network_spec: dict[str, dict],
                 global_networks: dict[str, nn.Module],
                 hyperparameters: dict[str, float]):
        self._action_space = env.action_space

        # Initialize networks
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
            self._local_networks[network] = self._local_networks[network]
        self._local_networks['controller'] = SharedActorCritic(
            shared_network=CustomModule(network_spec['controller_shared']),
            actor_network=CustomModule(network_spec['controller_actor']),
            critic_network=CustomModule(network_spec['controller_critic'])
        )
        self.sync_network()

        # Hyperparameters
        self._local_model_parameters = chain(*[self._local_networks[network].parameters() for network in self._networks])
        self._global_model_parameters = chain(*[self._global_networks[network].parameters() for network in self._networks])
        self._hyperparameters = hyperparameters
        self._optimizer = initialize_optimizer(self._hyperparameters['optimizer'],
                                               self._global_model_parameters,
                                               self._hyperparameters['learning_rate'])

        # Initialize memory data
        self._prev_action = torch.zeros(1, self._action_space.n)
        self._prev_inner_state = self._local_networks['inner_state_predictor'](torch.zeros(1, 260))

        self._batch_prev_observation = torch.zeros((1, 3, 64, 64))
        self._batch_prev_actions = torch.zeros((2, self._action_space.n))
        self._batch_prev_inner_state = self._local_networks['inner_state_predictor'](torch.zeros(1, 260))

    def get_action(self, 
                   observation: np.ndarray
                   ) -> np.ndarray:
        if self._hyperparameters['random_policy']:
            action = np.array(self._action_space.sample())
        else:
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
            
            self._prev_action = F.one_hot(action.detach(), num_classes=self._action_space.n).float()
        return action

    def sync_network(self) -> None:
        for network in self._networks:
            self._local_networks[network].load_state_dict(self._global_networks[network].state_dict())

    def _icm_module(self,
                    observations: Tensor,
                    actions: Tensor
                    ) -> tuple[Tensor, Tensor, float, float]:
        assert len(observations) == len(actions) + 1

        features = self._local_networks['feature_extractor'](observations)
        concatenated_features = torch.cat((features[:-1], features[1:]), dim=1)
        pred_actions = self._local_networks['inverse_network'](concatenated_features)
        inverse_loss = F.cross_entropy(pred_actions, actions)

        inverse_is_correct = torch.argmax(pred_actions, dim=1) == torch.argmax(actions, dim=1)
        inverse_accuracy = inverse_is_correct.sum() / len(inverse_is_correct)
        entropy = torch.sum(-pred_actions * torch.log(pred_actions + 1e-8), dim=1).mean().item()
        return features.detach(), inverse_loss, inverse_accuracy, entropy
    
    def _predictor_module(self,
                          features: Tensor,
                          actions: Tensor
                          ) -> tuple[Tensor, Tensor]:
        assert len(features) == len(actions)

        features_actions = torch.cat((features, actions), dim=1)
        inner_states = self._local_networks['inner_state_predictor'](features_actions, self._batch_prev_inner_state)
        self._batch_prev_inner_state = inner_states[-2].unsqueeze(dim=0).detach()

        inner_states_actions = torch.cat((inner_states[:-1], actions[1:]), dim=1)
        pred_features = self._local_networks['feature_predictor'](inner_states_actions)

        predictor_loss = F.mse_loss(pred_features, features[1:]) 
        return inner_states[1:].detach(), predictor_loss

    def _controller_module(self,
                           inner_states: Tensor,
                           actions: Tensor,
                           extrinsic_rewards: Tensor, 
                           intrinsic_rewards: Tensor
                           ) -> tuple[Tensor, Tensor, Tensor]:
        if self._hyperparameters['random_policy']:
            policy_loss = tensor(0.)
            value_loss = tensor(0.)
            entropy = tensor(0.)
        else:
            # GAE A2C
            assert len(actions) == len(extrinsic_rewards)
            assert len(extrinsic_rewards) == len(intrinsic_rewards)

            policy_loss = tensor(0.)
            value_loss = tensor(0.)
            entropy = tensor(0.)
            # rewards = extrinsic_rewards + self._hyperparameters['intrinsic_reward_scale'] * intrinsic_rewards
            # policies, v_preds = self._local_networks['controller'](inner_states)
            # distributions = Categorical(probs=policies)
            # log_probs = distributions.log_prob(actions)
            # entropy = distributions.entropy().mean()
            # policy_loss = -(gaes * log_probs).mean()
            # value_loss = F.mse_loss(v_preds, v_targets)
        return policy_loss, value_loss, entropy

    def train(self,
              batch: dict[str, Tensor]
              ) -> dict[str, float]:
        self._optimizer.zero_grad()

        # Initialize tensors
        observations = torch.cat((self._batch_prev_observation, batch['observations']), dim=0) # [t-1:T]
        actions = F.one_hot(batch['actions'], num_classes=self._action_space.n).float() # [t-2:T]
        actions = torch.cat((self._batch_prev_actions, actions), dim=0) # [t-2:T]

        # Calculate loss
        features, inverse_loss, inverse_accuracy, inverse_entropy = self._icm_module(observations, actions[1:-1])
        inner_states, predictor_loss = self._predictor_module(features, actions[:-1])
        predictor_loss = predictor_loss * 0. # Debug
        policy_loss, value_loss, controller_entropy = self._controller_module(inner_states,
                                                                              actions[2:],
                                                                              batch['extrinsic_rewards'],
                                                                              predictor_loss.detach())
        controller_loss = self._hyperparameters['policy_loss_scale'] * policy_loss\
                           + self._hyperparameters['value_loss_scale'] * value_loss\
                           + self._hyperparameters['entropy_scale'] * controller_entropy
        loss = self._hyperparameters['inverse_loss_scale'] * inverse_loss\
                + self._hyperparameters['predictor_loss_scale'] * predictor_loss\
                + controller_loss
        loss.backward()

        # Gradient clipping
        if self._hyperparameters['gradient_clipping'] != -1:
            nn.utils.clip_grad_norm_(self._local_model_parameters, self._hyperparameters['gradient_clipping'])

        # Update parameters with global parameters
        grad_norm = 0.
        for global_params, local_params in zip(self._global_model_parameters, self._local_model_parameters):
            global_params._grad = local_params.grad
            if local_params.grad is not None:
                grad_norm += (torch.norm(local_params.grad)**2).item()
        grad_norm = np.sqrt(grad_norm).item()
        self._optimizer.step()
        self.sync_network()
        
        # Update memory tensors
        self._batch_prev_observation = observations[-1].detach().unsqueeze(dim=0)
        self._batch_prev_actions = actions[-2:].detach()


        data = {}
        data['controller/entropy'] = controller_entropy.item()
        data['controller/policy_loss'] = policy_loss.item()
        data['controller/value_loss'] = value_loss.item()
        data['icm/inverse_acc'] = inverse_accuracy.item()
        data['icm/inverse_loss'] = inverse_loss.item()
        data['icm/predictor_loss'] = predictor_loss.item()
        data['icm/entropy'] = inverse_entropy
        data['optimizer/grad_norm'] = grad_norm
        return data