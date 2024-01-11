from datetime import datetime
from typing import Dict

from torch.utils.tensorboard import SummaryWriter

class LogWriter:
    def __init__(self, env_name: str=None, description: str='default'):
        formatted_time = datetime.now().strftime('%y%m%dT%H%M%S')
        self._writer = SummaryWriter(f'logs/{env_name}/{formatted_time}_{description}')

    def write(self, values: Dict[str, float], step: int) -> None:
        for value in values:
            self._writer.add_scalar(value, values[value], step)
        
    def close(self) -> None:
        self._writer.close()