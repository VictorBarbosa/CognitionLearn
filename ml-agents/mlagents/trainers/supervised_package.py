"""
Standalone package for supervised training in ML-Agents.
"""

__version__ = "1.0.0"

# Import main modules
from mlagents.trainers.supervised_settings import SupervisedLearningSettings
from mlagents.trainers.supervised_trainer import SupervisedTrainer
from mlagents.trainers.supervised_data_loader import SupervisedDataLoader
from mlagents.trainers.supervised_optimizer import SupervisedTorchOptimizer

# Import utilities
from mlagents.trainers.supervised_utils import (
    load_csv_data,
    prepare_observations,
    prepare_actions,
    split_data,
    augment_data
)

# Import configurations
# from mlagents.trainers.supervised_config import (
#     create_supervised_run_options,
#     DEFAULT_SUPERVISED_SETTINGS
# )

# Import CLI
from mlagents.trainers.mlagents_supervised import main as supervised_main

__all__ = [
    "SupervisedLearningSettings",
    "SupervisedTrainer",
    "SupervisedDataLoader",
    "SupervisedTorchOptimizer",
    "load_csv_data",
    "prepare_observations",
    "prepare_actions",
    "split_data",
    "augment_data",
    # "create_supervised_run_options",
    # "DEFAULT_SUPERVISED_SETTINGS",
    "supervised_main"
]
