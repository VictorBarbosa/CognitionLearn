# ## ML-Agent Learning (DrQv2)
# Contains the DrQv2Trainer class.

from typing import cast

from mlagents.trainers.sac.trainer import SACTrainer
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.drqv2.optimizer_torch import TorchDrQv2Optimizer, DrQv2Settings

TRAINER_NAME = "drqv2"


class DrQv2Trainer(SACTrainer):
    """
    The DrQv2Trainer is an implementation of the DrQ-v2 algorithm.
    It is a modification of SAC that uses data augmentation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hyperparameters: DrQv2Settings = cast(
            DrQv2Settings, self.trainer_settings.hyperparameters
        )

    def create_optimizer(self) -> TorchOptimizer:
        """
        Creates an Optimizer object.
        """
        return TorchDrQv2Optimizer(cast(TorchPolicy, self.policy), self.trainer_settings)

    @staticmethod
    def get_trainer_name() -> str:
        return TRAINER_NAME
