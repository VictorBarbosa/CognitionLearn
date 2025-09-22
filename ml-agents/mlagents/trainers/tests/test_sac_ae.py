import pytest
import attr

from mlagents.trainers.tests.simple_test_envs import (
    SimpleEnvironment,
)

from mlagents.trainers.sac_ae.settings import SACAESettings
from mlagents.trainers.settings import (
    NetworkSettings,
    TrainerSettings,
    RewardSignalSettings,
    RewardSignalType,
)
from mlagents.trainers.tests.check_env_trains import (
    check_environment_trains,
)

BRAIN_NAME = "1D"

# Configuração básica para SAC-AE
SAC_AE_CONFIG = TrainerSettings(
    trainer_type="sac_ae",
    hyperparameters=SACAESettings(
        learning_rate=3.0e-4,
        learning_rate_schedule="constant",
        batch_size=32,
        buffer_size=1000,
        buffer_init_steps=100,
        latent_size=128,
        ae_learning_rate=1e-3,
    ),
    network_settings=NetworkSettings(num_layers=1, hidden_units=64),
    summary_freq=100,
    max_steps=2000,
    threaded=False,
    reward_signals={RewardSignalType.EXTRINSIC: RewardSignalSettings()},
)


@pytest.mark.parametrize("action_sizes", [(0, 1), (1, 0)])
def test_simple_sac_ae(action_sizes):
    env = SimpleEnvironment([BRAIN_NAME], action_sizes=action_sizes)
    config = attr.evolve(SAC_AE_CONFIG)
    check_environment_trains(env, {BRAIN_NAME: config})


@pytest.mark.parametrize("action_sizes", [(0, 2), (2, 0)])
def test_2d_sac_ae(action_sizes):
    env = SimpleEnvironment([BRAIN_NAME], action_sizes=action_sizes, step_size=0.8)
    new_hyperparams = attr.evolve(
        SAC_AE_CONFIG.hyperparameters, buffer_init_steps=500
    )
    config = attr.evolve(
        SAC_AE_CONFIG, hyperparameters=new_hyperparams, max_steps=3000
    )
    check_environment_trains(env, {BRAIN_NAME: config})