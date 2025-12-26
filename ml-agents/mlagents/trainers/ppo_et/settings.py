from typing import Dict, List, NamedTuple, Optional, Union
import attr
import numpy as np
from mlagents.trainers.settings import (
    OnPolicyHyperparamSettings,
    ScheduleType,
)


@attr.s(auto_attribs=True)
class PPOETSettings(OnPolicyHyperparamSettings):
    beta: float = 5.0e-3
    epsilon: float = 0.2
    lambd: float = 0.95
    num_epoch: int = 3
    shared_critic: bool = False
    learning_rate_schedule: ScheduleType = ScheduleType.LINEAR
    beta_schedule: ScheduleType = ScheduleType.LINEAR
    epsilon_schedule: ScheduleType = ScheduleType.LINEAR
    # Parâmetros específicos do PPO-ET
    entropy_temperature: float = 1.0
    entropy_temperature_schedule: ScheduleType = ScheduleType.CONSTANT
    adaptive_entropy_temperature: bool = True
    target_entropy: Optional[float] = None