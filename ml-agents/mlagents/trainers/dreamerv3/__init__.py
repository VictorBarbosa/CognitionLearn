from mlagents.trainers.dreamerv3.trainer import DreamerV3Trainer, TRAINER_NAME
from mlagents.trainers.dreamerv3.settings import DreamerV3Settings

# This allows the plugin system to register the trainer
__all__ = ['DreamerV3Trainer', 'TRAINER_NAME', 'DreamerV3Settings']