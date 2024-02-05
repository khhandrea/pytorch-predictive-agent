from torch import nn, Tensor
from torch.nn.functional import normalize

class MLP(nn.Module):
    def __init__(self, 
                 layerwise_shape: tuple[int, ...],
                 activation: nn.Module = nn.ReLU,
                 normalize_input: bool = False,
                 end_with_softmax: bool = False):
        """
        Args:
            - layerwise_shape (tuple)
            - activation (nn.Module)
            - end_with_softmax (bool)
        """
        super().__init__()
        self._model = nn.Sequential()
        for idx in range(1, len(layerwise_shape) -1 ):
            self._model.add_module(
                f"layer{idx - 1}-linear",
                nn.Linear(layerwise_shape[idx - 1], layerwise_shape[idx])
            )
            self._model.add_module(
                f"layer{idx-1}-activation",
                activation()
            )            
        self._model.add_module(
            f"layer{len(layerwise_shape) - 1}-linear",
            nn.Linear(layerwise_shape[-2], layerwise_shape[-1])
        )
        if end_with_softmax:
            self._model.add_module(
                f"layer{len(layerwise_shape) - 1}-softmax",
                nn.Softmax(dim=1)
            )
        self._normalize_input = normalize_input

    def forward(self, input: Tensor) -> Tensor:
        if self._normalize_input:
            input = normalize(input, dim=0)
        output = self._model(input)
        return output