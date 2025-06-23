from typing import Any, Optional
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def get_checkpoint_path(name: str, checkpoint: Optional[str] = None) -> Path:
    if checkpoint is None:
        return max(Path.cwd().joinpath('checkpoints', name).glob('*'))
    return Path.cwd().joinpath('checkpoints', name, checkpoint)


def save_checkpoint(
    name: str,
    epoch: int,
    model: Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    scaler: Optional[GradScaler] = None
) -> str:
    root = Path.cwd().joinpath('checkpoints', name)
    root.mkdir(parents=True, exist_ok=True)
    states = {
        'epoch': epoch,
        'model': model.state_dict()
    }
    if optimizer is not None:
        states['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        states['scheduler'] = scheduler.state_dict()
    if scaler is not None:
        states['scaler'] = scaler.state_dict()

    p = root.joinpath(f'{epoch:0>3d}.pt')
    torch.save(states, p)
    return str(p)


def load_checkpoint_(
    f: Path,
    model: Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    scaler: Optional[GradScaler] = None,
    map_location: Any = 'cpu'
) -> int:
    c = torch.load(f, map_location=map_location, weights_only=True)
    if 'model' in c:
        model.load_state_dict(c['model'])
    else:
        model.load_state_dict(c)
        return -1

    if optimizer is not None and 'optimizer' in c:
        optimizer.load_state_dict(c['optimizer'])
    if scheduler is not None and 'scheduler' in c:
        scheduler.load_state_dict(c['scheduler'])
    if scaler is not None and 'scaler' in c:
        scaler.load_state_dict(c['scaler'])
    return c['epoch']
