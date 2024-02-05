from torch import nn, Tensor

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self._cnns = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self._fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[1:] == (3, 64, 64)
        x = self._cnns(x)
        print(x.shape)
        x = x.view(-1, 128 * 4 * 4)
        x = self._fc(x)
        return x