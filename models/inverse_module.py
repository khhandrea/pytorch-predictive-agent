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
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight.data)

    def forward(self, x: Tensor) -> Tensor:
        x = self._model(x)
        return x

class Layer3LinearInverseModule(nn.Module):
    def __init__(self):
        super().__init__()
        self._model = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Softmax(dim=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self._model(x)
        return x