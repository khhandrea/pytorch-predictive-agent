from torch import nn, Tensor

class DefaultDiscreteLinearActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self._shared = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self._actor = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Softmax(dim=1)
        )

        self._critic = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self._shared(x)
        policy = self._actor(x)
        value = self._critic(x)
        return policy, value