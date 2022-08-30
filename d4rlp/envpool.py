from typing import Dict, List, Tuple, Union

from dm_env.specs import Array, BoundedArray
import gym
from gym import spaces
import numpy as np
from tree import flatten_with_path


ENVPOOL_RESERVED = ('env_id', 'players')


def flatten(array) -> np.ndarray:
    return np.asarray(array).reshape(array.shape[0], -1)


def flatten_and_concat(envpool_timestep: List[Tuple[Tuple[str, ...], np.ndarray]]
                       ) -> np.ndarray:
    return np.hstack([flatten(obs)
                      for key, obs in envpool_timestep
                      if key[0] not in ENVPOOL_RESERVED])


def array2box(array: Union[Array, BoundedArray]) -> spaces.Box:
    if isinstance(array, BoundedArray):
        low, high = array.minimum, array.maximum
    else:
        low, high = -np.inf, np.inf
    low = np.broadcast_to(low, shape=array.shape)
    high = np.broadcast_to(high, shape=array.shape)
    return spaces.Box(low, high, dtype=array.dtype)


def spec2box(observation_spec) -> spaces.Box:
    lows, highs = [], []
    for key, spec in flatten_with_path(observation_spec):
        if key[0] in ENVPOOL_RESERVED:      # type: ignore
            continue
        parsed = array2box(spec)
        lows.append(np.asarray(parsed.low).ravel())
        highs.append(np.asarray(parsed.high).ravel())
    low, high = np.concatenate(lows), np.concatenate(highs)
    return spaces.Box(low=low, high=high, dtype=low.dtype)


class EnvPoolWrapper(gym.Env[np.ndarray, np.ndarray]):
    def __init__(self, env):
        self.env = env
        self.action_space: spaces.Box = array2box(env.action_spec())
        self.observation_space: spaces.Box = spec2box(env.observation_spec())

    def step(self, *args, **kwargs):
        timestep = self.env.step(*args, **kwargs)
        observation = flatten_with_path(timestep.observation)
        return (flatten_and_concat(observation),    # type: ignore
                timestep.reward,
                timestep.last(),
                {k[0]: v for k, v in observation})  # type: ignore

    def reset(self, *args, **kwargs):
        timestep = self.env.reset(*args, **kwargs)
        observation = flatten_with_path(timestep.observation)
        return flatten_and_concat(observation)      # type: ignore

    def __len__(self):
        return len(self.env)

    def __getattr__(self, name):
        return getattr(self.env, name)
