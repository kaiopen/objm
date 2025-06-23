from typing import Any, Dict, Optional, Sequence
import json
import os
import random

from tqdm import tqdm
import numpy as np
import torch
import torch.distributed
from torch.amp import GradScaler
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from kaitorch.utils import Configer, Logger

from src.data import DATASET
from src.model import Model
from .criterion import CRITERION
from .checkpoint import get_checkpoint_path, load_checkpoint_, save_checkpoint
from .optimizer import OPTIMIZER
from .scheduler import SCHEDULER


def process_logs(logs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def add(src: Dict[str, Any], dst: Dict[str, Any]) -> None:
        for k, v in src.items():
            if isinstance(v, Dict):
                add(v, dst[k])
            else:
                dst[k] += v

    def div(d: Dict[str, Any], div: float) -> None:
        for k, v in d.items():
            if isinstance(v, Dict):
                div(v, div)
            else:
                d[k] = v / div

    log = logs[0]
    for _log in logs[1:]:
        add(_log, log)

    div(log, len(logs))
    return log


class DDPTrainer:
    def __init__(self, cfg: Configer, *args, **kwargs) -> None:
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # NOTE: Keep this call or an open-many-files error occurs.
        # torch.multiprocessing.set_sharing_strategy('file_system')

        torch.distributed.init_process_group('nccl')

        rank = torch.distributed.get_rank()
        self._rank_0 = rank in (-1, 0)
        local_rank = int(os.environ['LOCAL_RANK'])
        self._local_rank_0 = local_rank in (-1, 0)
        torch.cuda.set_device(local_rank)

        self._name = cfg.name
        self._start_epoch = 0
        self._end_epoch = cfg.run.end_epoch

        # LOGGER
        self._logger = Logger(
            'train_ddp_' + self._name, level=Logger.INFO
        ) if self._local_rank_0 else lambda *args, **kwargs: None
        self._logger('\n' + str(cfg))

        # DATALOADER
        params = cfg.dataset.dict()
        self._dataset = DATASET[params.pop('type')](split='train', **params)
        self._sampler = DistributedSampler(self._dataset)
        self._loader = DataLoader(
            self._dataset,
            batch_size=cfg.run.batch_size,
            shuffle=False,
            sampler=self._sampler,
            num_workers=cfg.run.num_worker,
            collate_fn=self._dataset.collate,
            pin_memory=True,
            drop_last=True
        )
        self._num_batch = len(self._loader)
        self._num_acc = max(
            round(
                cfg.run.acc / (
                    cfg.run.batch_size * torch.distributed.get_world_size()
                )
            ),
            1
        )
        self._logger('=== The DATALOADER has been READY! ===')

        # MODEL
        self._model = Model(**cfg.model.dict())
        self._model = self._model.cuda()
        self._m = DistributedDataParallel(
            SyncBatchNorm.convert_sync_batchnorm(self._model),
            device_ids=[local_rank]
        )
        self._logger('=== The MODEL has been READY! ===')

        # CRITERION
        params = cfg.criterion.dict()
        self._criterion = CRITERION[params.pop('type')](**params)
        self._logger('=== The CRITERION has been READY! ===')

        # OPTIMIZER
        params = cfg.optimizer.dict()
        self._optimizer = OPTIMIZER[params.pop('type')](
            self._m.parameters(), **params
        )
        self._logger('=== The OPTIMIZER has been READY! ===')

        # SCHEDULER
        params = cfg.scheduler.dict()
        self._scheduler = SCHEDULER[params.pop('type')](
            self._optimizer, **params
        )
        self._logger('=== The SCHEDULER has been READY! ===')

        self._amp = cfg.run.amp
        if self._amp:
            self._scaler = GradScaler()
            self._logger('=== The AMP has been READY! ===')

        if cfg.run.resume:
            self._logger(
                '=== RESUMED checkpoint EPOCH'
                f' {self.resume(cfg.run.checkpoint)} ==='
            )

    def __call__(self):
        self._logger('\n=== TRAIN ===')
        self._m.train()
        for epoch in range(self._start_epoch, self._end_epoch):
            self._sampler.set_epoch(epoch)
            self.train_epoch(epoch)
        self._logger('\n=== DONE ===')
        del self._logger

    def resume(self, checkpoint: Optional[str] = None):
        epoch = load_checkpoint_(
            get_checkpoint_path(self._name, checkpoint),
            model=self._model,
            optimizer=self._optimizer,
            scheduler=self._scheduler
        )
        self._start_epoch = epoch + 1
        return epoch

    def save_checkpoint(self, epoch: int) -> str:
        return save_checkpoint(
            name=self._name,
            epoch=epoch,
            model=self._model,
            optimizer=self._optimizer,
            scheduler=self._scheduler
        )

    def train_epoch(self, epoch: int):
        loader = self._loader
        if self._local_rank_0:
            loader = tqdm(loader, desc=f'EPOCH {epoch:0>3d}')
            loader.set_postfix_str('loss=0.00000')
            logs = []

        self._optimizer.zero_grad()
        for it, (x, t) in enumerate(loader):
            if self._amp:
                with torch.autocast('cuda'):
                    loss, log = self._criterion(self._m(x.cuda()), t)
                self._scaler.scale(loss).backward()

                if 0 == (it + epoch * self._num_batch) % self._num_acc:
                    self._scaler.step(self._optimizer)
                    self._scaler.update()
                    self._optimizer.zero_grad()
            else:
                loss, log = self._criterion(self._m(x.cuda()), t)
                loss.backward()
                if 0 == (it + epoch * self._num_batch) % self._num_acc:
                    self._optimizer.step()
                    self._optimizer.zero_grad()
            if self._local_rank_0:
                loader.set_postfix_str(f'loss={loss.item():.5f}')
                logs.append(log)
        self._scheduler.step()

        if self._rank_0:
            self.save_checkpoint(epoch)
            self._logger(
                f'EPOCH {epoch}:\n' + json.dumps(process_logs(logs), indent=2)
            )
