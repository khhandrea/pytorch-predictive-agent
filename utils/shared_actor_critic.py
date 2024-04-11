from torch import nn, Tensor

class SharedActorCritic(nn.Module):
    def __init__(self,
                 shared_network: nn.Module,
                 actor_network: nn.Module,
                 critic_network: nn.Module
                 ) -> nn.Module:
        super().__init__()
        self._shared = shared_network
        self._actor = actor_network
        self._critic = critic_network

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = self._shared(x)
        policy = self._actor(z)
        value = self._critic(z)
        return policy, value.view(-1)