from typing import Dict

from torch.utils.tensorboard import SummaryWriter

class LogWriter:
    def __init__(self, name: str=None):
        self._writer = SummaryWriter(f'logs/{name}')

    def write(self, values: Dict[str, float], step: int) -> None:
        for value in values:
            self._writer.add_scalar(value, values[value], step)
        
    def close(self) -> None:
        self._writer.close()