from torch import nn, Tensor

class DiscreteLinearActorCritic(nn.Module):
    def __init__(self,
                 input_size: int,
                 action_space,
                 activation: nn.Module = nn.ReLU):
        super().__init__()
        self._action_size = action_space.n
        self._shared = nn.Sequential(
            nn.Linear(input_size, 128),
            activation(),
            nn.Linear(128, 64),
            activation()
        )

        self._actor = nn.Sequential(
            nn.Linear(64, 64),
            activation(),
            nn.Linear(64, self._action_size),
            nn.Softmax(dim=1)
        )

        self._critic = nn.Sequential(
            nn.Linear(64, 64),
            activation(),
            nn.Linear(64, 1)
        )

    def forward(self, input: Tensor) -> Tensor:
        z = self._shared(input)
        policy = self._actor(z)
        value = self._critic(z)
        return policy, value