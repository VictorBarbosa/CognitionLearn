from typing import Dict, List, NamedTuple, Optional, Union
import attr
import numpy as np
from mlagents.trainers.settings import (
    OnPolicyHyperparamSettings,
    ScheduleType,
)


@attr.s(auto_attribs=True)
class PPOCESettings(OnPolicyHyperparamSettings):
    # Parâmetros do PPO padrão
    beta: float = 5.0e-3
    epsilon: float = 0.2
    lambd: float = 0.95
    num_epoch: int = 3
    shared_critic: bool = False
    learning_rate_schedule: ScheduleType = ScheduleType.LINEAR
    beta_schedule: ScheduleType = ScheduleType.LINEAR
    epsilon_schedule: ScheduleType = ScheduleType.LINEAR
    
    # Parâmetros específicos do PPO-CE
    curiosity_strength: float = 0.01
    curiosity_gamma: float = 0.99
    curiosity_learning_rate: float = 3e-4
    curiosity_hidden_units: int = 128
    curiosity_num_layers: int = 2
    imagination_horizon: int = 5
    use_imagination_augmented: bool = True
    curiosity_loss_weight: float = 1.0