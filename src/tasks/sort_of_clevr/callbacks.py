from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ...core.callbacks import BaseCallBack, CallbackSpec, accuracy
from ...core.config import CallbackConfig
from ...core.contracts import VQABatch, VQAOutput
from .spec import (
    Q_TYPES_OFFSET, Q_TYPE_IDX, SUBTYPE_NAMES, SUB_Q_TYPE_IDX,
)

if TYPE_CHECKING:
    from torch import Tensor

    from ...core.config import Config


@dataclass
class AccuracyCBCfg(CallbackConfig):
    name: str = 'accuracy'
    subtypes: bool = True # report the nine subtypes and three families acc


class AccuracyCB(BaseCallBack):

    def __init__(self, subtypes: bool = True) -> None:
        super().__init__()
        self.subtypes = subtypes

    def _groups(self, questions: Tensor) -> dict[str, Tensor]:
        groups = {
            family: questions[:, Q_TYPE_IDX + offset] == 1
            for family, offset in Q_TYPES_OFFSET.items()
        }
        if self.subtypes:
            for (family, subtype), sname in SUBTYPE_NAMES.items():
                fam = questions[:, Q_TYPE_IDX + Q_TYPES_OFFSET[family]] == 1
                sub = questions[:, SUB_Q_TYPE_IDX + subtype] == 1
                groups[sname] = fam & sub
        return groups

    def metrics(self, out: VQAOutput, batch: VQABatch) -> dict[str, float]:
        logits, answers = out['logits'], batch['answers']
        metrics = {'accuracy': accuracy(logits, answers)}
        metrics.update({
            f'{name}_accuracy': accuracy(logits[mask], answers[mask])
            for name, mask in self._groups(batch['questions']).items()
        })
        return metrics

    def on_train_step_end(
            self, trainer, out: VQAOutput, batch: VQABatch,
            ) -> Optional[dict[str, float]]:
        return self.metrics(out, batch)

    def on_eval_step_end(
            self, trainer, out: VQAOutput, batch: VQABatch,
            ) -> Optional[dict[str, float]]:
        return self.metrics(out, batch)

    def on_test_step_end(
            self, trainer, out: VQAOutput, batch: VQABatch,
            ) -> Optional[dict[str, float]]:
        return self.metrics(out, batch)

    @classmethod
    def from_config(
            cls, cfg: Config, cb_cfg: AccuracyCBCfg,
            ) -> AccuracyCB:
        return cls(subtypes=bool(cb_cfg.subtypes))


CALLBACKS: dict[str, CallbackSpec] = {
    'accuracy': CallbackSpec(AccuracyCBCfg, AccuracyCB),
}


