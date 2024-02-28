from torch import nn, Tensor

class DefaultCNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self._cnns = nn.Sequential(
            # 3 x 64 x 64
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 16 x 32 x 32
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 32 x 16 x 16
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 64 x 8 x 8
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 128 x 4 x 4
        )

        self._fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[1:] == (3, 64, 64)
        x = self._cnns(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self._fc(x)
        return x

class CuriosityLikeCNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self._cnns = nn.Sequential(
            # 3 x 64 x 64
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            # 32 x 32 x 32
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            
            # 32 x 16 x 16
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            # 32 x 8 x 8
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            # 16 x 4 x 4 (256)
        )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight.data)

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[1:] == (3, 64, 64)
        x = self._cnns(x)
        x = x.view(-1, 16 * 4 * 4)
        return x