import torch
from torch.distributions.categorical import Categorical
from torch import nn, optim, Tensor

from models import DiscreteLinearActorCritic

class ControllerAgent:
    def __init__(self,
                 action_space, 
                 random_policy: bool, 
                 device: torch.device,
                 gamma: float,
                 controller_lr: float,
                 feature_size: int,
                 controller_network_layerwise_shape: tuple[int, ...]
                 ):
        self._action_space = action_space
        self._random_policy = random_policy
        self._device = device
        self._actor_critic = DiscreteLinearActorCritic(
            controller_network_layerwise_shape, 
            action_space=action_space).to(self._device)
        self._gamma = gamma

        self._loss_mse = nn.MSELoss()
        self._prev_input = torch.zeros(1, feature_size).to(self._device)
        self._log_prob = torch.tensor(0).to(self._device)
        self._controller_optimizer = optim.Adam(self._actor_critic.parameters(), lr=controller_lr)

    def get_action_and_update(self, input: Tensor, reward: float) -> tuple[Tensor, float, float]:
        if self._random_policy:
            policy_loss = 0.
            value_loss = 0.
            random_action = self._action_space.sample()
            action = torch.tensor(random_action, device=self._device).unsqueeze(0)
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