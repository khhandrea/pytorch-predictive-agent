import numpy as np
import torch
from torch import Tensor, tensor

class OnPolicyExperienceReplay():
    """
    On-policy experience replay. items: observations, actions, extrinsic_rewards, dones
    """

    def __init__(self):
        self.size = 0
        self._keys = ('observations', 'actions', 'extrinsic_rewards', 'dones')
        self._reset()

    def _reset(self):
        for k in self._keys:
            setattr(self, k, None)
        self.size = 0

    def add_experience(self, *experience) -> None:
        """
        Add one step data

        Attributes:
            observation(numpy.ndarray): observation
            action(int): discrete action
            extrinsic_reward(float): extrinsic reward
            dones(bool): mask whether the episode is end
        """
        assert len(experience) == len(self._keys)

        for idx, key in enumerate(self._keys):
            new_item = np.expand_dims(np.array(experience[idx]).squeeze(), 0)
            if getattr(self, key) is not None:
                new_item = np.concatenate((getattr(self, key), new_item))
            setattr(self, key, new_item)
        self.size += 1

    def sample(self,
               to_tensor: bool = False
               ) -> dict[str, Tensor | np.ndarray]:
        """
        Sample from the on-policy experience replay

        Attributes:
            to_tensor(bool): if true, the types of all of the return items will be Tensor

        Returns:
            batch(dict[str, Tensor | numpy.ndarray]): dictionary of observations, actions, extrinsic_rewards, dones
        """
        batch = {}
        for key in self._keys:
            batch[key] = getattr(self, key)
            if to_tensor:
                batch[key] = tensor(batch[key], dtype=torch.float32)
        self._reset()
        return batch