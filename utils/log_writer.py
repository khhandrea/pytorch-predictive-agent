from datetime import datetime
from typing import Dict

from torch.utils.tensorboard import SummaryWriter

class LogWriter:
    def __init__(self, 
                 env_name: str=None, 
                 description: str='default',
                 skip_log: bool=False):
        self._skip_log = skip_log
        if self._skip_log:
            return
        
        formatted_time = datetime.now().strftime('%y%m%dT%H%M%S')
        self._writer = SummaryWriter(f'logs/{env_name}/{formatted_time}_{description}')

    def write(self, values: Dict[str, float], step: int) -> None:
        if self._skip_log:
            return
        
        for value in values:
            self._writer.add_scalar(value, values[value], step)
        
    def close(self) -> None:
        if self._skip_log:
            return
        self._writer.close()