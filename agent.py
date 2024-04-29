from itertools import chain

import numpy as np
import torch
from torch import nn, Tensor, tensor
from torch.distributions.categorical import Categorical
from torch.nn import functional as F

from utils import CustomModule, SharedActorCritic
from utils import initialize_optimizer, calc_returns, calc_gaes

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

        # Parameters
        global_model_parameters = chain(*[self._global_networks[network].parameters() for network in self._networks])
        self._hyperparameters = hyperparameters
        self._global_optimizer = initialize_optimizer(self._hyperparameters['optimizer'],
                                                      global_model_parameters,
                                                      self._hyperparameters['learning_rate'])

        # Initialize memory data
        self._prev_action = torch.zeros(1, self._action_space.n)
        self._prev_inner_state = self._local_networks['inner_state_predictor'](torch.zeros(1, 260))

        self._batch_prev_observation = torch.zeros((1, 3, 64, 64))
        self._batch_prev_actions = torch.zeros((2, self._action_space.n))
        self._batch_prev_inner_state = self._local_networks['inner_state_predictor'](torch.zeros(1, 260))

    def sync_network(self) -> None:
        for network in self._networks:
            self._local_networks[network].load_state_dict(self._global_networks[network].state_dict())

    def get_action(self, 
                   observation: np.ndarray
                   ) -> int:
        if self._hyperparameters['random_policy']:
            action = np.array(self._action_space.sample())
        else:
            with torch.no_grad():
                observation = torch.from_numpy(observation).float().unsqueeze(0)

                # Feature extractor
                feature = self._local_networks['feature_extractor'](observation)

                # Inner state predictor
                feature_action = torch.cat((feature, self._prev_action), dim=1)
                inner_state = self._local_networks['inner_state_predictor'](feature_action, self._prev_inner_state)
                self._prev_inner_state = inner_state[-1:].detach()

                # Controller
                policy = self._local_networks['controller'].policy(inner_state)
                action = Categorical(probs=policy).sample()
                
                self._prev_action = F.one_hot(action, num_classes=self._action_space.n).float()
        return action

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

    def _get_advantages_v_targets(self,
                                  rewards: Tensor,
                                  batch: dict[str, Tensor],
                                  v_preds: Tensor
                                  ) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            last_feature = self._local_networks['feature_extractor'](batch['observations'][-1:])
            last_v_pred = self._local_networks['controller'].value(last_feature)
        if self._hyperparameters.get('lmbda'):
            v_preds_all = torch.cat((v_preds, last_v_pred), dim=0)
            advantages = calc_gaes(rewards, batch['dones'], v_preds_all, self._hyperparameters['gamma'], self._hyperparameters['lmbda'])
            v_target = advantages + v_preds
        else:
            returns = calc_returns(rewards, batch['dones'], last_v_pred, self._hyperparameters['gamma'])
            advantages = returns - v_preds
            v_target = returns
        return advantages, v_target

    def _controller_module(self,
                           inner_states: Tensor,
                           actions: Tensor,
                           batch: dict[str, Tensor],
                           intrinsic_rewards: Tensor
                           ) -> tuple[Tensor, Tensor, Tensor]:
        if self._hyperparameters['random_policy']:
            policy_loss = tensor(0.)
            value_loss = tensor(0.)
            entropy = np.log(actions.shape[-1])
        else:
            rewards = batch['extrinsic_rewards'] + self._hyperparameters['intrinsic_reward_scale'] * intrinsic_rewards
            policies = self._local_networks['controller'].policy(inner_states)
            v_preds = self._local_networks['controller'].value(inner_states)
            advantages, v_targets = self._get_advantages_v_targets(rewards, batch, v_preds.detach())
            distributions = Categorical(policies)
            actions = torch.argmax(actions, dim=1)
            log_probs = distributions.log_prob(actions)
            entropy = distributions.entropy().mean()
            policy_loss = -(advantages * log_probs).mean()
            value_loss = F.mse_loss(v_preds, v_targets)
        return policy_loss, value_loss, entropy

    def _update_global_parameters(self) -> float:
        grad_norm = 0.
        global_model_parameters = chain(*[self._global_networks[network].parameters() for network in self._networks])
        local_model_parameters = chain(*[self._local_networks[network].parameters() for network in self._networks])

        # Gradient clipping
        if self._hyperparameters['gradient_clipping'] != -1:
            norm = nn.utils.clip_grad_norm_(local_model_parameters, self._hyperparameters['gradient_clipping'])

        for global_params, local_params in zip(global_model_parameters, local_model_parameters):
            global_params._grad = local_params.grad
            # Calculate gradient norm
            if local_params.grad is not None:
                grad_norm += torch.norm(local_params.grad)**2
            # Reset local gradients
            local_params.grad = None
        grad_norm = np.sqrt(grad_norm).item()
        self._global_optimizer.step()
        self.sync_network()

        return grad_norm

    def train(self,
              batch: dict[str, Tensor]
              ) -> dict[str, float]:
        self._global_optimizer.zero_grad()

        # Initialize tensors
        observations = torch.cat((self._batch_prev_observation, batch['observations']), dim=0) # [t-1:T]
        actions = F.one_hot(batch['actions'].to(torch.int64), num_classes=self._action_space.n).float() # [t-2:T]
        actions = torch.cat((self._batch_prev_actions, actions), dim=0) # [t-2:T]

        # Calculate loss
        features, inverse_loss, inverse_accuracy, inverse_entropy = self._icm_module(observations, actions[1:-1])
        inner_states, predictor_loss = self._predictor_module(features, actions[:-1])
        policy_loss, value_loss, controller_entropy = self._controller_module(inner_states,
                                                                              actions[2:],
                                                                              batch,
                                                                              predictor_loss.detach())
        controller_loss = self._hyperparameters['policy_loss_scale'] * policy_loss\
                           + self._hyperparameters['value_loss_scale'] * value_loss\
                           + self._hyperparameters['entropy_scale'] * controller_entropy
        loss = self._hyperparameters['inverse_loss_scale'] * inverse_loss\
                + self._hyperparameters['predictor_loss_scale'] * predictor_loss\
                + controller_loss
        loss.backward()


        grad_norm = self._update_global_parameters()
        
        # Update memory tensors
        self._batch_prev_observation = observations[-1].unsqueeze(dim=0)
        self._batch_prev_actions = actions[-2:]

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