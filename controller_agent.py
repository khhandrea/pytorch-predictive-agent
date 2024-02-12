import torch
from torch.distributions.categorical import Categorical
from torch import nn, optim, Tensor

from utils import get_class_from_module, makedir_and_save_module, get_load_path

class ControllerAgent:
    def __init__(self,
                 feature_size: int,
                 action_space, 
                 random_policy: bool, 
                 controller_module: str,
                 gamma: float,
                 controller_lr: float,
                 policy_discount: float,
                 device: torch.device,
                 path: str,
                 ):
        self._action_space = action_space
        self._random_policy = random_policy
        self._device = device
        self._controller = get_class_from_module('models', controller_module)().to(self._device)
        self._gamma = gamma
        self._policy_discount = policy_discount
        self._path = path

        self._loss_mse = nn.MSELoss()
        self._prev_input = torch.zeros(1, feature_size).to(self._device)
        self._log_prob = torch.tensor(0).to(self._device)
        self._controller_optimizer = optim.Adam(self._controller.parameters(), lr=controller_lr)

    def get_action_and_update(self, input: Tensor, reward: float) -> tuple[Tensor, float, float]:
        if self._random_policy:
            policy_loss = 0.
            value_loss = 0.
            entropy = 0.
            random_action = self._action_space.sample()
            action = torch.tensor(random_action, device=self._device).unsqueeze(0)
        else:
            # Update
            self._controller_optimizer.zero_grad()
            policy, value = self._controller(input)
            _, prev_value = self._controller(self._prev_input)
            advantage = reward + self._gamma * value - prev_value
            policy_loss_tensor = -advantage * self._log_prob
            value_loss_tensor = self._loss_mse(reward + self._gamma * value, prev_value)
            loss = self._policy_discount * policy_loss_tensor + value_loss_tensor
            loss.backward()
            self._controller_optimizer.step()
            self._prev_input = input.detach()

            # Get an action
            distribution = Categorical(probs=policy)
            action = distribution.sample()
            self._log_prob = distribution.log_prob(action).detach()

            policy_loss = policy_loss_tensor.item()
            value_loss = value_loss_tensor.item()
            entropy = distribution.entropy().item()
        return action, policy_loss, value_loss, entropy
    
    def save(self, description: str):
        makedir_and_save_module(
            self._controller.state_dict(),
            self._path,
            'controller-network',
            description)
        
    def load(self, path: str):
        self._controller.load_state_dict(
            torch.load(get_load_path(path, 'controller-network')))