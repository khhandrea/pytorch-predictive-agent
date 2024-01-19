from typing import Dict

from torch.utils.tensorboard import SummaryWriter

class LogWriter:
    def __init__(self, 
                 path: str,
                 skip_log: bool):
        self._skip_log = skip_log
        if self._skip_log:
            return
        
        self._writer = SummaryWriter(f'logs/{path}')

    def write(self, values: Dict[str, float], step: int) -> None:
        if self._skip_log:
            return
        
        for value in values:
            self._writer.add_scalar(value, values[value], step)
        
    def close(self) -> None:
        if self._skip_log:
            return
        self._writer.close()