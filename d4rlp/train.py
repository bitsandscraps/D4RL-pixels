from pathlib import Path
import pprint
import random
from typing import List, Tuple, Union

import envpool
import numpy as np
from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.policy import SACPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import BaseLogger, LazyLogger, TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
import torch
from torch import Tensor
import torch.backends.cudnn
from torch.optim import Optimizer

from .envpool import EnvPoolWrapper
from .utils import ArgumentParser, Logger


def build_argument_parser():
    parser = ArgumentParser()
    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--alpha-lr', type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--buffer-size', type=int, default=1000000)
    parser.add_argument('--critic-lr', type=float, default=3e-4)
    parser.add_argument('--epoch', type=int, default=3000)
    parser.add_argument('--epoch-medium', type=int, default=1000)
    parser.add_argument('--fix-alpha', action='store_false', dest='auto_alpha')
    parser.add_argument('--frame-skip', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[256, 256])
    parser.add_argument('--max-episode-steps', type=int, default=1000)
    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--start-timesteps', type=int, default=1000)
    parser.add_argument('--step-per-collect', type=int, default=1000)
    parser.add_argument('--step-per-epoch', type=int, default=1)
    parser.add_argument('--task', default='WalkerRun-v1')
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--update-per-step', type=int, default=1)
    return parser


def create_env(task: str,
               training_num: int,
               test_num: int,
               seed: int,
               frame_skip: int,
               max_episode_steps: int):
    train_envs = envpool.make_dm(task,
                                 num_envs=training_num,
                                 seed=seed,
                                 frame_skip=frame_skip,
                                 max_episode_steps=max_episode_steps)
    test_envs = envpool.make_dm(task,
                                num_envs=test_num,
                                seed=seed + training_num,
                                frame_skip=frame_skip,
                                max_episode_steps=max_episode_steps)
    return EnvPoolWrapper(train_envs), EnvPoolWrapper(test_envs)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(actor_lr: float,
          alpha: float,
          alpha_lr: float,
          auto_alpha: float,
          batch_size: int,
          buffer_size: int,
          critic_lr: float,
          device: torch.device,
          epoch: int,
          epoch_medium: int,
          frame_skip: int,
          gamma: float,
          hidden_sizes: List[int],
          logger: Logger,
          max_episode_steps: int,
          n_step: int,
          seed: int,
          start_timesteps: int,
          step_per_collect: int,
          step_per_epoch: int,
          task: str,
          tau: float,
          test_num: int,
          training_num: int,
          update_per_step: int,
          ):
    set_seed(seed)
    train_envs, test_envs = create_env(task=task,
                                       training_num=training_num,
                                       test_num=test_num,
                                       seed=seed,
                                       frame_skip=frame_skip,
                                       max_episode_steps=max_episode_steps)
    state_shape = train_envs.observation_space.shape
    assert state_shape is not None
    action_shape = train_envs.action_space.shape
    assert action_shape is not None
    max_action = train_envs.action_space.high[0]
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

    policy = SACPolicy(actor,
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

    # collector
    if training_num > 1:
        buffer = VectorReplayBuffer(buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(buffer_size)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    train_collector.collect(n_step=start_timesteps, random=True)

    save_path: Path = Path(__file__).resolve().parent / 'data' / task
    save_path.mkdir(exist_ok=True, parents=True)

    def save_checkpoint_fn(epoch_: int, env_step: int, gradient_step: int):
        del env_step, gradient_step
        if epoch_ == epoch_medium - 1:
            torch.save(policy.state_dict(), save_path / 'medium.pth')

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
    torch.save(policy.state_dict(), save_path / 'expert.pth')

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
