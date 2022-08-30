import envpool
import numpy as np
from tianshou.data import Batch

from d4rlp.envpool import EnvPoolWrapper


def test_wrapper():
    env = envpool.make_dm('WalkerRun-v1', num_envs=10)
    wrapper = EnvPoolWrapper(env)
    assert len(wrapper) == 10
    assert not wrapper.is_async
    obs = wrapper.reset()
    assert obs.shape == (10, 24)
    obs, rew, done, info = wrapper.step(np.ones((10, 6)))
    assert obs.shape == (10, 24)
    assert rew.shape == (10,)
    assert done.shape == (10,)
    assert done.dtype == bool
    assert isinstance(info, dict)
    Batch(obs=obs, rew=rew, done=done, info=info)
    assert True
