import numpy as np
import torch
from torch import Tensor, tensor

class OnPolicyExperienceReplay():
    def __init__(self):
        self.size = 0
        self._keys = ('observations', 'actions', 'extrinsic_rewards', 'dones')
        self._reset()

    def _reset(self):
        for k in self._keys:
            setattr(self, k, None)
        self.size = 0

    def add_experience(self, *experience) -> None:
        assert len(experience) == len(self._keys)

        for idx, key in enumerate(self._keys):
            new_item = np.expand_dims(np.array(experience[idx]).squeeze(), 0)
            if getattr(self, key) is not None:
                new_item = np.concatenate((getattr(self, key), new_item))
            setattr(self, key, new_item)
        self.size += 1

    def sample(self, to_tensor: bool) -> dict[str, Tensor | np.ndarray]:
        batch = {}
        for key in self._keys:
            batch[key] = getattr(self, key)
            if to_tensor:
                batch[key] = tensor(batch[key], dtype=torch.float32)
        self._reset()
        return batch