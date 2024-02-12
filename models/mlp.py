from torch import nn, Tensor

class CuriosityLinearInverseModule(nn.Module):
    def __init__(self):
        super().__init__()
        self._model = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Softmax(dim=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self._model(x)
        return x

class CuriosityLinearFeaturePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self._model = nn.Sequential(
            nn.Linear(256 + 4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self._model(x)
        return x