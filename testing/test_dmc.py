from dm_control.suite import ALL_TASKS
from gym.utils.env_checker import check_env
import pytest

from d4rlp.dmc import DMCEnv


@pytest.mark.parametrize('domain,task', ALL_TASKS)
def test_dmcenv(domain, task):
    env = DMCEnv(domain, task)
    check_env(env)
    env.seed(0)
