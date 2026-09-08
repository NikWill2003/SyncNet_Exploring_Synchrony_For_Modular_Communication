from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from ...core.callbacks import BaseCallBack, CallbackSpec
from ...core.config import CallbackConfig
from .model import SyncNet

if TYPE_CHECKING:
    from ...core.config import Config


class FrozenStep(nn.Module):
    def forward(self, z: Tensor, h: Tensor) -> Tensor:
        return z


class FrozenPhase(SyncNet):
    def build_dynamics(self, cfg): return FrozenStep()


class ZeroPhase(FrozenPhase):
    def initial_phases(self, z_slots, B):
        z = torch.zeros(B, self.N, self.d, device=z_slots.device, dtype=z_slots.dtype)
        z[..., 0] = 1.0
        return z


class ShuffledPhase(SyncNet):
    def initial_phases(self, z_slots, B):
        return super().initial_phases(z_slots[:, torch.randperm(self.M, device=z_slots.device)], B)


def as_(model: SyncNet, view: type[SyncNet]) -> SyncNet:

    base = type(model)
    cls = view if base is SyncNet else type(f'{view.__name__}Of{base.__name__}', (view, base), {})
    m = cls(model.cfg, model.dataset, model.answer_dim).to(next(model.parameters()).device)
    m.load_state_dict(model.state_dict(), strict=False)            
    return m.eval()


# base for all overrides apart from the test time T ablation
@dataclass
class OverrideCfg(CallbackConfig):
    max_batches: int = 0                                          


class Override(BaseCallBack):
    view: type[SyncNet]
    key: str

    def __init__(self, max_batches: int = 0) -> None:
        self.max_batches = max_batches

    @classmethod
    def from_config(cls, cfg: Config, cb_cfg: OverrideCfg):
        return cls(max_batches=int(cb_cfg.max_batches))

    def on_train_end(self, trainer) -> None:
        model = trainer.unwrapped_model()
        if not isinstance(model, SyncNet) or isinstance(model, self.view):
            return
        base = trainer.evaluate(trainer.test_dataloader, 'test', max_batches=self.max_batches)['callbacks/accuracy']
        hit = trainer.evaluate(trainer.test_dataloader, 'test', model=as_(model, self.view), max_batches=self.max_batches)['callbacks/accuracy']
        trainer.summary({f'override/{self.key}_drop': base - hit}, 'test')


class FreezePhaseOverride(Override):    key, view = 'freeze_phase', FrozenPhase
class ZeroPhaseOverride(Override):      key, view = 'zero_phase', ZeroPhase
class ShufflePhaseOverride(Override):   key, view = 'shuffle_phase', ShuffledPhase


OVERRIDES: dict[str, CallbackSpec] = {
    c.key: CallbackSpec(OverrideCfg, c, requires=frozenset({'sync'}))
    for c in (FreezePhaseOverride, ZeroPhaseOverride, ShufflePhaseOverride)
}
