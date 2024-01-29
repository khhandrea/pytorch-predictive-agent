from torch import nn, Tensor

class DiscreteLinearActorCritic(nn.Module):
    def __init__(self,
                 layerwise_shape: tuple[int, ...],
                 action_space,
                 activation: nn.Module = nn.ReLU):
        super().__init__()
        self._action_size = action_space.n
        self._shared = nn.Sequential()
        for idx in range(1, len(layerwise_shape) -1 ):
            self._shared.add_module(
                f"layer{idx - 1}-linear",
                nn.Linear(layerwise_shape[idx - 1], layerwise_shape[idx])
            )
            self._shared.add_module(
                f"layer{idx-1}-activation",
                activation()
            )            
        self._shared.add_module(
            f"layer{len(layerwise_shape) - 1}-linear",
            nn.Linear(layerwise_shape[-2], layerwise_shape[-1])
        )

        self._actor = nn.Sequential(
            nn.Linear(layerwise_shape[-1], layerwise_shape[-1]),
            activation(),
            nn.Linear(layerwise_shape[-1], self._action_size),
            nn.Softmax(dim=1)
        )

        self._critic = nn.Sequential(
            nn.Linear(layerwise_shape[-1], layerwise_shape[-1]),
            activation(),
            nn.Linear(layerwise_shape[-1], 1)
        )

    def forward(self, input: Tensor) -> Tensor:
        z = self._shared(input)
        policy = self._actor(z)
        value = self._critic(z)
        return policy, value