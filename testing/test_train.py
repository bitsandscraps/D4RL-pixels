from pathlib import Path
from typing import List

from dm_control.suite import ALL_TASKS
import gym
from gym.utils.env_checker import check_env
import h5py
import numpy as np
import pytest

from d4rlp.dmc import env_name
from d4rlp.train import ReplayWrapper


@pytest.mark.parametrize('domain,task', ALL_TASKS)
def test_check_wrapper(domain, task):
    env = gym.make(env_name(domain, task))
    check_env(ReplayWrapper(env, 1000000))


@pytest.mark.parametrize('samples', [1, 2, 5, 10])
def test_check_replay(samples, tmp_path: Path):
    env = gym.make(env_name('walker', 'walk'),
                   max_episode_steps=4)
    env = ReplayWrapper(env, samples, tmp_path)
    action_space = env.action_space
    action_space.seed(0)
    index = -1
    for index in range((samples + 3) // 4):
        states: List[np.ndarray] = []
        pixels: List[np.ndarray] = []
        rewards: List[float] = []
        done = False
        state = env.reset()
        for _ in range(4):
            assert not done
            states.append(state)                # type: ignore
            pixels.append(env.env.render())     # type: ignore
            state, reward, done, _ = env.step(action_space.sample())
            rewards.append(reward)
        assert done
        with h5py.File(tmp_path / f'trajectory{index}.hdf5', 'r') as file:
            np.testing.assert_allclose(file['state'][...], np.stack(states))
            np.testing.assert_allclose(file['pixels'][...], np.stack(pixels))
            np.testing.assert_allclose(file['reward'][...], rewards)
            np.testing.assert_equal(file['done'][...],
                                    [False, False, False, True])
    assert not (tmp_path / f'trajectory{index + 1}.hdf5').exists()
