from torch import nn, Tensor

class DefaultLSTMInnerStatePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self._model = nn.LSTM(
            input_size = 4 + 256,
            hidden_size = 256,
            num_layers = 1
        )

    def forward(self, x: Tensor) -> Tensor:
        inner_state, _ = self._model(x)
        return inner_state