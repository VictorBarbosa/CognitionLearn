# ## ML-Agent Learning (DCAC)
# Contains the DCACTrainer class.

from typing import cast

from mlagents.trainers.sac.trainer import SACTrainer
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.dcac.optimizer_torch import TorchDCACOptimizer, DCACSettings

TRAINER_NAME = "dcac"


class DCACTrainer(SACTrainer):
    """
    The DCACTrainer is an implementation of the DCAC algorithm.
    It is a modification of SAC that checks for destructive critic updates.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hyperparameters: DCACSettings = cast(
            DCACSettings, self.trainer_settings.hyperparameters
        )

    def create_optimizer(self) -> TorchOptimizer:
        """
        Creates an Optimizer object.
        """
        return TorchDCACOptimizer(cast(TorchPolicy, self.policy), self.trainer_settings)

    @staticmethod
    def get_trainer_name() -> str:
        return TRAINER_NAME