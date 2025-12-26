# ## ML-Agent Learning (DrQv2)
# Contains an implementation of DrQv2 as described in https://arxiv.org/abs/2107.09645
# This is a modification of the SAC implementation, with data augmentation.

from typing import cast, Dict

import attr
from mlagents.torch_utils import torch

from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.trajectory import ObsUtil
from mlagents.trainers.sac.optimizer_torch import TorchSACOptimizer, SACSettings
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.settings import TrainerSettings
from mlagents.trainers.torch_entities.utils import ModelUtils


@attr.s(auto_attribs=True)
class DrQv2Settings(SACSettings):
    """
    DrQv2-specific hyperparameters.
    """

    image_pad: int = 4


class TorchDrQv2Optimizer(TorchSACOptimizer):
    """
    This is a modification of the SAC optimizer that uses data augmentation (random shifts)
    on visual observations, as described in the DrQ-v2 paper.
    """

    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        super().__init__(policy, trainer_settings)
        self.hyperparameters: DrQv2Settings = cast(
            DrQv2Settings, trainer_settings.hyperparameters
        )
        self.image_pad = self.hyperparameters.image_pad

    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        """
        Updates model using buffer. Overrides the SAC optimizer to apply data augmentation.
        :param batch: Experience mini-batch.
        :param num_sequences: Number of trajectories in batch.
        :return: Output from update process.
        """
        # Apply data augmentation (random shifts)
        augmented_batch = self._apply_augmentation(batch)
        return super().update(augmented_batch, num_sequences)

    def _apply_augmentation(self, batch: AgentBuffer) -> AgentBuffer:
        """
        Applies random shifts to the visual observations in the batch.
        """
        n_obs = len(self.policy.behavior_spec.observation_specs)
        if n_obs == 0:
            return batch

        # Get the visual observation indices
        vis_obs_indices = [
            i
            for i, spec in enumerate(self.policy.behavior_spec.observation_specs)
            if len(spec.shape) == 3
        ]

        if not vis_obs_indices:
            return batch

        # Create a copy of the batch to modify
        # This is not the most memory-efficient, but it is the safest.
        augmented_batch = batch.copy()

        for i in vis_obs_indices:
            obs_key = ObsUtil.get_name_at(i)
            next_obs_key = ObsUtil.get_name_at_next(i)

            obs = ModelUtils.list_to_tensor(augmented_batch[obs_key])
            next_obs = ModelUtils.list_to_tensor(augmented_batch[next_obs_key])

            # The same random shift should be applied to obs and next_obs
            augmented_obs, augmented_next_obs = self._random_shift(
                obs, next_obs, self.image_pad
            )

            augmented_batch[obs_key].set(ModelUtils.to_numpy(augmented_obs))
            augmented_batch[next_obs_key].set(ModelUtils.to_numpy(augmented_next_obs))

        return augmented_batch

    @staticmethod
    def _random_shift(
        obs: torch.Tensor, next_obs: torch.Tensor, pad: int
    ) -> (torch.Tensor, torch.Tensor):
        """
        Applies the same random shift to a batch of observations and next_observations.
        :param obs: A batch of observations, with shape (N, C, H, W)
        :param next_obs: A batch of next observations, with shape (N, C, H, W)
        :param pad: The amount of padding to apply to each side.
        :return: A tuple of augmented obs and next_obs.
        """
        n, c, h, w = obs.shape
        device = obs.device
        
        # Add padding
        padded_obs = torch.nn.functional.pad(obs, (pad, pad, pad, pad), mode="replicate")
        padded_next_obs = torch.nn.functional.pad(
            next_obs, (pad, pad, pad, pad), mode="replicate"
        )

        # Generate random shifts for the entire batch at once
        rand_h = torch.randint(0, 2 * pad + 1, (n, 1, 1, 1), device=device, dtype=torch.long)
        rand_w = torch.randint(0, 2 * pad + 1, (n, 1, 1, 1), device=device, dtype=torch.long)

        # Generate coordinate grids
        h_coords = torch.arange(h, device=device).view(1, 1, h, 1).expand(n, 1, h, 1)
        w_coords = torch.arange(w, device=device).view(1, 1, 1, w).expand(n, 1, 1, w)
        
        # Add the random shifts
        h_indices = h_coords + rand_h
        w_indices = w_coords + rand_w

        # Expand indices to cover all channels
        h_indices = h_indices.expand(n, c, h, w)
        w_indices = w_indices.expand(n, c, h, w)

        # Generate channel indices
        c_indices = torch.arange(c, device=device).view(1, c, 1, 1).expand(n, c, h, w)

        # Use advanced indexing to extract the shifted patches
        # For the batch dimension
        batch_indices = torch.arange(n, device=device).view(n, 1, 1, 1).expand(n, c, h, w)

        # Apply the shifts using advanced indexing
        augmented_obs = padded_obs[batch_indices, c_indices, h_indices, w_indices]
        augmented_next_obs = padded_next_obs[batch_indices, c_indices, h_indices, w_indices]

        return augmented_obs, augmented_next_obs