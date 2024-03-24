import numpy as np
from torch import Tensor, tensor

class OnPolicyExperienceReplay():
    def __init__(self):
        self.size = 0
        self._keys = ('observations', 'actions', 'extrinsic_rewards', 'dones')
        self._reset()

    def _reset(self):
        for k in self._keys:
            setattr(self, k, [])
        self.size = 0

    def add_experience(
            self, 
            observation: np.ndarray,
            action: int,
            extrinsic_reward: float,
            done: bool
        ) -> None:
        most_recent = (observation, action, extrinsic_reward, done)
        for idx, key in enumerate(self._keys):
            getattr(self, key).append(most_recent[idx])
        self.size += 1

    def sample(self) -> dict[str, Tensor]:
        batch = {}
        for key in self._keys:
            if key == 'dones':
                setattr(self, key, np.array((getattr(self, key)), dtype=np.float32))
            batch[key] = tensor(np.array(getattr(self, key)))
        self._reset()
        return batch