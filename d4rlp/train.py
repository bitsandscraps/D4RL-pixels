from collections import defaultdict
from pathlib import Path
import pprint
import random
from typing import Dict, List, Optional, Tuple, Union

import gym
import h5py
import numpy as np
from tianshou.data import Collector, ReplayBuffer
from tianshou.env import BaseVectorEnv, DummyVectorEnv, ShmemVectorEnv
from tianshou.policy import SACPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import BaseLogger, LazyLogger, TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
import torch
from torch import Tensor
import torch.backends.cudnn
from torch.optim import Optimizer

from d4rlp.dmc import env_name
from d4rlp.utils import ArgumentParser, Logger


class ReplayWrapper(gym.Wrapper):
    def __init__(self,
                 env: gym.Env,
                 samples: int,
                 save_dir: Optional[Path] = None):
        super().__init__(env, new_step_api=False)
        self.samples = samples
        self.trajectory_index = 0
        self.save_dir = save_dir
        self.done = False
        self.buffer: Dict[str, List[Union[np.ndarray, float, bool]]]
        self.buffer = defaultdict(list)

    def step(self, action):
        state, reward, done, info = super().step(action)    # type: ignore
        self.buffer['reward'].append(reward)
        if not done:
            self.buffer['state'].append(state.copy())
            self.render()
        elif self.save_dir is not None:
            trajectory_state = np.stack(self.buffer['state'])
            trajectory_reward = np.asarray(self.buffer['reward'])
            trajectory_pixels = np.stack(self.buffer['pixels'])
            trajectory_done = np.zeros_like(trajectory_reward, dtype=bool)
            trajectory_done[-1] = True
            path = self.save_dir / f'trajectory{self.trajectory_index}.hdf5'
            with h5py.File(path, 'w') as file:
                file.create_dataset('state', data=trajectory_state)
                file.create_dataset('reward', data=trajectory_reward)
                file.create_dataset('pixels', data=trajectory_pixels)
                file.create_dataset('done', data=trajectory_done)
            self.trajectory_index += 1
            self.samples -= trajectory_done.size
            if self.samples <= 0:
                self.save_dir = None
        self.done = done
        return state, reward, done, info

    def reset(self, **kwargs):
        self.buffer = defaultdict(list)
        if kwargs.get('return_info', False):
            state, info = super().reset(**kwargs)
        else:
            state = super().reset(**kwargs)
            info = None
        assert isinstance(state, np.ndarray)
        self.buffer['state'].append(state.copy())
        self.render()
        return state if info is None else (state, info)

    def render(self, *args, **kwargs):
        pixels = super().render(*args, **kwargs)
        assert isinstance(pixels, np.ndarray)
        self.buffer['pixels'].append(pixels)

    @property
    def render_mode(self):
        return None

def build_argument_parser():
    parser = ArgumentParser()
    parser.add_argument('--actor-lr', type=float, default=1e-3)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--alpha-lr', type=float, default=3e-4)
    parser.add_argument('--auto-alpha', action='store_true')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--buffer-size', type=int, default=1000000)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--domain', default='walker')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--frame-skip', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[256, 256])
    parser.add_argument('--max-episode-steps', type=int, default=1000)
    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--samples', type=int, default=1000000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--start-timesteps', type=int, default=10000)
    parser.add_argument('--step-per-collect', type=int, default=1)
    parser.add_argument('--step-per-epoch', type=int, default=5000)
    parser.add_argument('--task', default='walk')
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--update-per-step', type=int, default=1)
    return parser


def build_policy(actor_lr: float,
                 alpha: float,
                 alpha_lr: float,
                 auto_alpha: bool,
                 critic_lr: float,
                 device: torch.device,
                 gamma: float,
                 hidden_sizes: List[int],
                 n_step: int,
                 tau: float,
                 train_envs: BaseVectorEnv,
                 ) -> SACPolicy:
    state_shape = train_envs.observation_space.shape
    assert state_shape is not None
    action_shape = train_envs.action_space.shape
    assert action_shape is not None
    max_action = train_envs.action_space.high[0]    # type: ignore
    print('Observations shape:', state_shape)
    print('Actions shape:', action_shape)
    # model
    net_a = Net(state_shape, hidden_sizes=hidden_sizes, device=device)
    actor = ActorProb(net_a,
                      action_shape,
                      max_action=max_action,
                      device=device,
                      unbounded=True,
                      conditioned_sigma=True).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    net_c1 = Net(state_shape,
                 action_shape,
                 hidden_sizes=hidden_sizes,
                 concat=True,
                 device=device)
    net_c2 = Net(state_shape,
                 action_shape,
                 hidden_sizes=hidden_sizes,
                 concat=True,
                 device=device)
    critic1 = Critic(net_c1, device=device).to(device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=critic_lr)
    critic2 = Critic(net_c2, device=device).to(device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=critic_lr)

    alpha_: Union[Tuple[Tensor, Tensor, Optimizer], float] = alpha
    if auto_alpha:
        target_entropy = -np.prod(action_shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=alpha_lr)
        alpha_ = (target_entropy, log_alpha, alpha_optim)

    return SACPolicy(actor,
                     actor_optim,
                     critic1,
                     critic1_optim,
                     critic2,
                     critic2_optim,
                     tau=tau,
                     gamma=gamma,
                     alpha=alpha_,
                     estimation_step=n_step,
                     action_space=train_envs.action_space)


def create_env(task: str,
               test_num: int,
               seed: int,
               frame_skip: int,
               max_episode_steps: int,
               samples: int,
               save_dir: Optional[Path]):
    env = gym.make(task,
                   frame_skip=frame_skip,
                   max_episode_steps=max_episode_steps)
    env_ = ReplayWrapper(env, samples=samples, save_dir=save_dir)
    train_envs = DummyVectorEnv([lambda: env_])
    test_envs = ShmemVectorEnv(
        [lambda: gym.make(task,
                          frame_skip=frame_skip,
                          max_episode_steps=max_episode_steps)
         for _ in range(test_num)])
    train_envs.seed(seed)
    test_envs.seed(seed)
    return train_envs, test_envs


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(actor_lr: float,
          alpha: float,
          alpha_lr: float,
          auto_alpha: bool,
          batch_size: int,
          buffer_size: int,
          critic_lr: float,
          device: torch.device,
          domain: str,
          epoch: int,
          frame_skip: int,
          gamma: float,
          hidden_sizes: List[int],
          logger: Logger,
          max_episode_steps: int,
          n_step: int,
          samples: int,
          seed: int,
          start_timesteps: int,
          step_per_collect: int,
          step_per_epoch: int,
          task: str,
          tau: float,
          test_num: int,
          update_per_step: int,
          ):
    set_seed(seed)
    env_name_ = env_name(domain, task)

    save_dir: Optional[Path] = None
    if logger.root is not None:
        save_dir = Path(__file__).resolve().parents[1] / 'data' / env_name_
        save_dir.mkdir(exist_ok=True, parents=True)

    if save_dir is None:
        replay_save_dir: Optional[Path] = None
        policy_save_dir: Optional[Path] = None
    else:
        replay_save_dir = save_dir / 'replay'
        policy_save_dir = save_dir / 'policy'
        replay_save_dir.mkdir(exist_ok=True)
        policy_save_dir.mkdir(exist_ok=True)

    train_envs, test_envs = create_env(task=env_name_,
                                       test_num=test_num,
                                       seed=seed,
                                       frame_skip=frame_skip,
                                       max_episode_steps=max_episode_steps,
                                       samples=samples,
                                       save_dir=replay_save_dir)
    policy = build_policy(actor_lr=actor_lr,
                          alpha=alpha,
                          alpha_lr=alpha_lr,
                          auto_alpha=auto_alpha,
                          critic_lr=critic_lr,
                          device=device,
                          gamma=gamma,
                          hidden_sizes=hidden_sizes,
                          n_step=n_step,
                          tau=tau,
                          train_envs=train_envs)

    # collector
    buffer = ReplayBuffer(buffer_size)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    train_collector.collect(n_step=start_timesteps, random=True)

    def save_checkpoint_fn(epoch_: int, env_step: int, gradient_step: int):
        del env_step, gradient_step
        if policy_save_dir is not None:
            torch.save(policy.state_dict(),
                       policy_save_dir / f'epoch{epoch_}.pth')

    if logger.root is None:
        tianshou_logger: BaseLogger = LazyLogger()
    else:
        tianshou_logger = TensorboardLogger(logger.writer)

    # trainer
    result = offpolicy_trainer(policy,
                               train_collector,
                               test_collector,
                               epoch,
                               step_per_epoch,
                               step_per_collect,
                               test_num,
                               batch_size,
                               save_checkpoint_fn=save_checkpoint_fn,
                               logger=tianshou_logger,
                               update_per_step=update_per_step,
                               test_in_train=False)
    pprint.pprint(result)

    # Let's watch its performance!
    policy.eval()
    test_envs.seed(seed)
    test_collector.reset()
    result = test_collector.collect(n_episode=test_num)
    mean_reward = result['rews'].mean()
    mean_length = result['lens'].mean()
    print(f'Final reward: {mean_reward}, length: {mean_length}')


if __name__ == '__main__':
    train(**vars(build_argument_parser().parse_args()))
