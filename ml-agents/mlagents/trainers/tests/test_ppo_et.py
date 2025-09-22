import pytest
import attr

from mlagents.trainers.tests.simple_test_envs import (
    SimpleEnvironment,
)

from mlagents.trainers.ppo_et.settings import PPOETSettings
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

# Configuração básica para PPO-ET
PPO_ET_CONFIG = TrainerSettings(
    trainer_type="ppo_et",
    hyperparameters=PPOETSettings(
        learning_rate=5.0e-3,
        learning_rate_schedule="constant",
        batch_size=16,
        buffer_size=64,
        entropy_temperature=1.0,
        adaptive_entropy_temperature=True,
    ),
    network_settings=NetworkSettings(num_layers=1, hidden_units=32),
    summary_freq=500,
    max_steps=3000,
    threaded=False,
    reward_signals={RewardSignalType.EXTRINSIC: RewardSignalSettings()},
)


@pytest.mark.parametrize("action_sizes", [(0, 1), (1, 0)])
def test_simple_ppo_et(action_sizes):
    env = SimpleEnvironment([BRAIN_NAME], action_sizes=action_sizes)
    config = attr.evolve(PPO_ET_CONFIG)
    check_environment_trains(env, {BRAIN_NAME: config})


@pytest.mark.parametrize("action_sizes", [(0, 2), (2, 0)])
def test_2d_ppo_et(action_sizes):
    env = SimpleEnvironment([BRAIN_NAME], action_sizes=action_sizes, step_size=0.8)
    new_hyperparams = attr.evolve(
        PPO_ET_CONFIG.hyperparameters, batch_size=64, buffer_size=640
    )
    config = attr.evolve(
        PPO_ET_CONFIG, hyperparameters=new_hyperparams, max_steps=10000
    )
    check_environment_trains(env, {BRAIN_NAME: config})