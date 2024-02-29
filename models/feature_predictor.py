from torch import nn, Tensor

class CuriosityLinearFeaturePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        linear1 = nn.Linear(256 + 4, 256)
        linear2 = nn.Linear(256, 256)
        self._model = nn.Sequential(
            linear1,
            nn.ELU(),
            linear2,
            nn.ELU()
        )

        nn.init.kaiming_uniform_(linear1.weight)
        nn.init.xavier_uniform_(linear2.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self._model(x)
        return x