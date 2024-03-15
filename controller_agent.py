import torch
from torch.distributions.categorical import Categorical
from torch import nn, optim, Tensor, tensor

class ControllerAgent:
    def __init__(self,
                 feature_size: int,
                 action_space, 
                 random_policy: bool, 
                 controller_module: str,
                 gamma: float,
                 lr: float,
                 optimizer_arg: str,
                 policy_discount: float,
                 entropy_discount: float,
                 ):
        self._prev_input = torch.zeros(1, feature_size).to(self._device)
        self._log_prob = tensor(0).to(self._device)

    def get_action_and_update(self, input: Tensor, reward: float) -> tuple[Tensor, float, float]:
        # Update
        policy, value = self._controller(input)
        _, prev_value = self._controller(self._prev_input)
        distribution = Categorical(probs=policy)

        advantage = reward + self._gamma * value - prev_value
        value_loss = self._loss_mse(reward + self._gamma * value, prev_value)
        policy_loss = -advantage * self._log_prob
        entropy = distribution.entropy()
        loss = value_loss + self._policy_discount * policy_loss + self._entropy_discount * entropy

        self._prev_input = input.detach()

        # Get an action
        action = distribution.sample()
        self._log_prob = distribution.log_prob(action).detach()