from torch import nn, Tensor

class CuriosityLinearFeaturePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self._model = nn.Sequential(
            nn.Linear(256 + 4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight.data)

    def forward(self, x: Tensor) -> Tensor:
        x = self._model(x)
        return x