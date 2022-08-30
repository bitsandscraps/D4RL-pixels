import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Final, Optional
import warnings

import tomlkit
import torch
from torch.utils.tensorboard.writer import SummaryWriter

import __main__


MAIN: Final[str] = 'main'
ROOT = Path(__file__).resolve().parents[1] / 'results'


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self,
                 *args,
                 append: bool = False,
                 parent: bool = False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if append and not parent:
            warnings.warn('Setting parent to True because append is True.')
            parent = True
        self.append = append
        self.parent = parent
        self.add_defaults()

    def add_argument(self, *args, **kwargs):
        action = super().add_argument(*args, **kwargs)
        if action.dest == MAIN:
            raise argparse.ArgumentError(action, f'{MAIN} is a reserved keyword')
        return action

    def add_defaults(self):
        self.add_argument('--config', default='')
        group = self.add_mutually_exclusive_group()
        group.add_argument('--cuda', type=int, default=0)
        group.add_argument('--cpu', action='store_const', dest='cuda', const=-1)
        self.add_argument('-q', '--no-log', action='store_false', dest='log')
        if self.append:
            self.add_argument('path')

    def parse_args(self, args=None, namespace=None):
        arguments = super().parse_args(args, namespace)
        if arguments.config:
            with Path(arguments.config).open() as file:
                defaults = tomlkit.load(file)
            self.set_defaults(**defaults)
        arguments = super().parse_args(args, namespace)
        if self.append:
            arguments.path = Path(arguments.path)
        root: Optional[Path]
        if self.parent:
            if not (arguments.path / 'arguments.toml').is_file():
                raise ValueError(f'Not a log directory: {arguments.path}')
            parent: Optional[Logger] = Logger(root=arguments.path, parent=None)
        else:
            parent = None
        if arguments.log:
            now = datetime.now()
            arguments.timestamp = now.isoformat()
            if self.append:
                root = arguments.path
            else:
                root = ROOT
            root = root / str(now.date()) / str(now.time())
        else:
            root = None
        if not torch.cuda.is_available():
            arguments.cuda = -1
        logger = Logger(root=root, parent=parent)
        if root is not None:
            print('Logging to', root)
        setattr(arguments, MAIN, __main__.__spec__.name)
        logger.save_args(arguments)
        arguments.logger = logger
        if arguments.cuda == -1:
            arguments.device = torch.device('cpu')
        else:
            arguments.device = torch.device(f'cuda:{arguments.cuda}')
        dict_args = vars(arguments)
        for key in ('config', 'cuda', 'log', MAIN, 'path', 'timestamp'):
            dict_args.pop(key, None)
        return arguments


class DummyWriter(SummaryWriter):
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def add_scalars(self, *args, **kwargs):
        pass

    def add_histogram(self, *args, **kwargs):
        pass

    def add_image(self, *args, **kwargs):
        pass

    def add_images(self, *args, **kwargs):
        pass

    def add_figure(self, *args, **kwargs):
        pass

    def add_video(self, *args, **kwargs):
        pass

    def add_audio(self, *args, **kwargs):
        pass

    def add_text(self, *args, **kwargs):
        pass

    def add_graph(self, *args, **kwargs):
        pass

    def add_embedding(self, *args, **kwargs):
        pass

    def add_pr_curve(self, *args, **kwargs):
        pass

    def add_custom_scalars(self, *args, **kwargs):
        pass

    def add_mesh(self, *args, **kwargs):
        pass

    def add_hparams(self, *args, **kwargs):
        pass

    def flush(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass


class Logger:
    def __init__(self, root: Optional[Path], parent: Optional["Logger"]) -> None:
        self.root = root
        self.parent = parent
        self._writer: Optional[SummaryWriter] = None

    @property
    def writer(self) -> SummaryWriter:
        if self._writer is None:
            if self.root is None:
                self._writer = DummyWriter()
            else:
                self._writer = SummaryWriter(log_dir=self.root)
        return self._writer

    def save_args(self, args: argparse.Namespace) -> None:
        dict_args = vars(args)
        if self.root is None:
            tomlkit.dump(dict_args, sys.stdout)
        else:
            with (self.root / 'arguments.toml').open('w') as file:
                tomlkit.dump(dict_args, file)
