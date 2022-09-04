from collections import deque
from itertools import starmap
from typing import Optional, Union

from dm_control import suite
from dm_control.rl.control import Environment
from dm_env import TimeStep
from dm_env.specs import Array, BoundedArray
import gym
from gym import spaces
from gym.utils.renderer import Renderer
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
    metadata = {'render_modes': ['rgb_array', 'single_rgb_array']}

    def __init__(self,
                 domain: str,
                 task: str,
                 render_mode: Optional[str] = None,
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
        self.render_mode = render_mode
        self.renderer = Renderer(render_mode, self._render)
        timestep = self.env._n_sub_steps * self.env.physics.timestep()
        timestep *= frame_skip
        self.metadata['render_fps'] = 1 / timestep

    def step(self, action):
        timestep: Optional[TimeStep] = None
        reward = 0
        for _ in range(self.frame_skip):
            timestep = self.env.step(action)
            reward += timestep.reward
        self.renderer.render_step()
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

    def render(self, mode='single_rgb_array'):
        if self.render_mode is not None:
            return self.renderer.get_renders()
        else:
            return self._render(mode)

    def _render(self, mode='single_rgb_array'):
        assert mode in self.metadata['render_modes']
        return self.env.physics.render(camera_id=self.camera_id,
                                       width=self.width,
                                       height=self.height)

    def seed(self, seed: Optional[int] = None):
        self.env: Environment = suite.load(
            self.domain, self.task,
            task_kwargs={'random': seed},
            environment_kwargs={'flat_observation': True})


deque(starmap(register, suite.ALL_TASKS), maxlen=0)
