from torch import nn, Tensor

class DiscreteLinearActorCritic(nn.Module):
    def __init__(self,
                 input_size: int,
                 action_size: int,
                 activation: nn.Module = nn.ReLU):
        super().__init__()
        self._shared = nn.Sequential(
            nn.Linear(input_size, 128),
            activation(),
            nn.Linear(128, 64),
            activation()
        )

        self._actor = nn.Sequential(
            nn.Linear(64, 64),
            activation(),
            nn.Linear(64, action_size),
            nn.Softmax(dim=1)
        )

        self._critic = nn.Sequential(
            nn.Linear(64, 64),
            activation(),
            nn.Linear(64, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self._shared(x)
        policy = self._actor(x)
        value = self._critic(x)
        return policy, value
    
class LSTMActorCritic(nn.Module):
    def __init__(self,
                 input_size: int,
                 action_size: int,
                 activation: nn.Module = nn.ReLU):
        super().__init__()
        self._shared = nn.Sequential()
        self._actor = nn.Sequential()
        self._critic = nn.Sequential()

    def forward(self, x: Tensor) -> Tensor:
        x = self._shared
        policy = self._actor(x)
        value = self._critic(x)
        return policy, value