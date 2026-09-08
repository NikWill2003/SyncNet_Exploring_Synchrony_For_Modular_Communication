from __future__ import annotations

import itertools
import time

from .logging import MultiAverageMeter
from .trainer import Trainer
from ..utils import get_batch_dict_size


class DebugTrainer(Trainer):

    def on_train_start(self) -> None:
        self.train_meter = MultiAverageMeter()

        self.batch = next(iter(self.train_dataloader))
        self.batch_size = get_batch_dict_size(self.batch)
        self.train_batch_iter = itertools.repeat(self.batch)

        self.total_step = 0
        self.opt_step = 0
        self.should_stop = False

        self.train_init_log()
        self.log_info(
            f'DEBUG | one batch of {self.batch_size} | {self.train_cfg.n_steps} steps'
        )

        self.time_cuda_sync()
        self.step_interval_start = time.perf_counter()
        self.tot_training_time = 0

    def on_eval_hit(self) -> None:
        pass

    def on_train_end(self) -> dict[str, float]:
        averages = self.train_meter.get_averages()
        return {'loss': float(averages.get('loss', float('nan')))}
