from typing import Dict, List, NamedTuple, Optional, Union
import attr
import numpy as np
from mlagents.trainers.settings import (
    OffPolicyHyperparamSettings,
    ScheduleType,
)


@attr.s(auto_attribs=True)
class SACAESettings(OffPolicyHyperparamSettings):
    # Parâmetros do SAC padrão
    batch_size: int = 128
    buffer_size: int = 50000
    buffer_init_steps: int = 0
    tau: float = 0.005
    steps_per_update: float = 1
    save_replay_buffer: bool = False
    init_entcoef: float = 1.0
    reward_signal_steps_per_update: float = attr.ib()
    
    @reward_signal_steps_per_update.default
    def _reward_signal_steps_per_update_default(self):
        return self.steps_per_update
    
    # Parâmetros específicos do SAC-AE
    latent_size: int = 512
    ae_learning_rate: float = 1e-3
    ae_hidden_units: int = 256
    ae_num_layers: int = 2
    world_model_learning_rate: float = 3e-4
    world_model_hidden_units: int = 256
    world_model_num_layers: int = 2
    use_autoencoder: bool = True
    use_world_model: bool = True
    ae_loss_weight: float = 1.0
    world_model_loss_weight: float = 1.0
    reconstruction_loss_weight: float = 1.0