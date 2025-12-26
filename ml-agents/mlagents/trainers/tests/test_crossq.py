import pytest
import attr


from mlagents.trainers.tests.simple_test_envs import (
    SimpleEnvironment,
)

from mlagents.trainers.tests.dummy_config import (
    crossq_dummy_config,
)
from mlagents.trainers.tests.check_env_trains import (
    check_environment_trains,
)

BRAIN_NAME = "1D"

CROSSQ_TORCH_CONFIG = crossq_dummy_config()

# tests in this file won't be tested on GPU machine
pytestmark = pytest.mark.slow


@pytest.mark.parametrize("action_sizes", [(0, 1), (1, 0)])
def test_simple_crossq(action_sizes):
    env = SimpleEnvironment([BRAIN_NAME], action_sizes=action_sizes)
    config = attr.evolve(CROSSQ_TORCH_CONFIG)
    check_environment_trains(env, {BRAIN_NAME: config})
