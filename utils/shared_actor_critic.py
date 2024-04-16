from torch import nn, Tensor

class SharedActorCritic(nn.Module):
    """
    Actor-critic pytorch module neural network with shared body.
    """
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
    
    def policy(self, x: Tensor) -> Tensor:
        z = self._shared(x)
        policy = self._actor(z)
        return policy
    
    def value(self, x: Tensor) -> Tensor:
        z = self._shared(x)
        value = self._critic(z)
        return value.view(-1)