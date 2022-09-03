from collections import deque
from itertools import starmap
from typing import Optional, Union

from dm_control import suite
from dm_control.rl.control import Environment
from dm_env import TimeStep
from dm_env.specs import Array, BoundedArray
import gym
from gym import spaces
import numpy as np


def array2box(array: Union[Array, BoundedArray]) -> spaces.Box:
    if isinstance(array, BoundedArray):
        low, high = array.minimum, array.maximum
    else:
        low, high = -np.inf, np.inf
    low = np.broadcast_to(low, shape=array.shape)
    high = np.broadcast_to(high, shape=array.shape)
    return spaces.Box(low, high, dtype=array.dtype)


def env_name(domain: str, task: str) -> str:
    # HumainoidCMU is an exception
    words = [w.capitalize() if w != 'CMU' else w for w in domain.split('_')]
    words.extend([w.capitalize() for w in task.split('_')])
    return ''.join(words) + '-v1'


def register(domain: str, task: str) -> None:
    gym.register(id=env_name(domain, task),
                 entry_point=DMCEnv,
                 kwargs={'domain': domain, 'task': task})


class DMCEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {'render_modes': ['single_rgb_array']}
    render_mode = 'single_rgb_array'

    def __init__(self,
                 domain: str,
                 task: str,
                 frame_skip: int = 1,
                 camera_id: int = 0,
                 width: int = 64,
                 height: int = 64):
        self.domain = domain
        self.task = task
        self.seed()
        self.action_space = array2box(self.env.action_spec())
        observation_spec = self.env.observation_spec()
        self.observation_space = array2box(observation_spec['observations'])
        self.frame_skip = frame_skip
        self.camera_id = camera_id
        self.width = width
        self.height = height

    def step(self, action):
        timestep: Optional[TimeStep] = None
        reward = 0
        for _ in range(self.frame_skip):
            timestep = self.env.step(action)
            reward += timestep.reward
        assert timestep is not None
        return timestep.observation['observations'], reward, False, False, {}

    def reset(self, *,
              seed: Optional[int] = None,
              return_info: bool = False,
              options: Optional[dict] = None):
        super().reset(seed=seed, return_info=return_info, options=options)
        if seed is not None:
            self.seed(seed)
        observation = self.env.reset().observation['observations']
        if return_info:
            return observation, {}
        else:
            return observation

    def render(self):
        return self.env.physics.render(camera_id=self.camera_id,
                                       width=self.width,
                                       height=self.height)

    def seed(self, seed: Optional[int] = None):
        self.env: Environment = suite.load(
            self.domain, self.task,
            task_kwargs={'time_limit': float('inf'), 'random': seed},
            environment_kwargs={'flat_observation': True})


deque(starmap(register, suite.ALL_TASKS), maxlen=0)
