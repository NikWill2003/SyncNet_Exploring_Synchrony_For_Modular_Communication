from __future__ import annotations

from typing import Callable, Iterator

import torch
import torch.nn as nn
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader

from .config import Config
from .callbacks import CallBackList

OnDeviceIter = Iterator[dict[str, torch.Tensor]]
Loaders = tuple[DataLoader | OnDeviceIter, ...]
LossFn = Callable[[dict, dict], tuple[Tensor, dict[str, float]]]


def register_configs() -> None:
    from ..tasks import DATASETS
    from ..models.baselines import MODELS
    from ..models.syncnet import SyncNetConfig

    cs = ConfigStore.instance()

    cs.store(name='config_schema', node=Config)

    for dataset in DATASETS.values():
        cs.store(group='dataset', name=f'{dataset.NAME}_base',
                 node=dataset.DATA_CONFIG)

    for name, (config, _) in MODELS.items():
        cs.store(group='model', name=f'{name}_base', node=config)
    cs.store(group='model', name='syncnet_base', node=SyncNetConfig)


def get_dataset(cfg: Config):
    from ..tasks import DATASETS

    dataset = DATASETS.get(cfg.dataset.name)
    if dataset is None:
        raise ValueError(
            f'unknown dataset: {cfg.dataset.name!r} (available: {sorted(DATASETS)})'
        )
    return dataset


def build_model(cfg: Config) -> nn.Module:
    from ..models.baselines import MODELS

    name = str(cfg.model.name)
    if name == 'syncnet':
        from ..models.syncnet import ABLATIONS
        dataset = get_dataset(cfg)
        ablation = cfg.model.ablation or None  # type: ignore
        if ablation not in ABLATIONS:
            raise ValueError(f'unknown ablation: {ablation!r} (available: {sorted(k for k in ABLATIONS if k)})')
        return ABLATIONS[ablation](cfg.model, dataset.NAME, int(dataset.ANSWER_DIM))  # type: ignore
    entry = MODELS.get(name)
    if entry is None:
        raise ValueError(
            f'unknown model: {name!r} (available: {sorted(MODELS)})'
        )
    _, model_class = entry
    dataset = get_dataset(cfg)
    return model_class.from_config(
        cfg.model, dataset.NAME, int(dataset.ANSWER_DIM)
        )


def build_callbacks(cfg: Config, model=None):

    from .callbacks import SHARED_CALLBACKS
    from ..models.syncnet import SYNC_CALLBACKS

    dataset = get_dataset(cfg)
    available = {**SHARED_CALLBACKS, **getattr(dataset, 'CALLBACKS', {}), **SYNC_CALLBACKS}
    offered = frozenset(getattr(model, 'supported_callbacks', ()))

    wanted = [{'name': 'accuracy'}] + [({'name': c} if isinstance(c, str) else dict(c)) for c in (cfg.callbacks or [])]
    callbacks = []
    for cb_cfg in wanted:
        spec = available.get(cb_cfg['name'])
        if spec is None:
            raise ValueError(f"unknown callback {cb_cfg['name']!r}; available: {sorted(available)}")
        missing = spec.requires - offered
        if missing:
            raise ValueError(f"callback {cb_cfg['name']!r} requires {sorted(missing)}, but {type(model).__name__} supports {sorted(offered)}")
        typed = OmegaConf.merge(OmegaConf.structured(spec.config), cb_cfg)
        callbacks.append(spec.callback_class.from_config(cfg, typed))

    return CallBackList(callbacks)


def build_dataloaders(cfg: Config, device: str) -> Loaders:

    name = str(cfg.dataset.name)
    if name == 'sort_of_clevr':
        from ..tasks.sort_of_clevr.loader import build_dataloaders as _build
    elif name == 'sqoop':
        from ..tasks.sqoop.loader import build_dataloaders as _build
    else:
        raise ValueError(f'unknown dataset: {name!r}')
    return _build(cfg, device) # type: ignore


def build_loss_fn(cfg: Config) -> LossFn:

    ce_loss = nn.CrossEntropyLoss()
    
    def cross_entropy(
        out: dict, batch: dict,
        ) -> tuple[Tensor, dict[str, float]]:

        loss = ce_loss(out['logits'].float(), batch['answers'])
        return loss, {'loss': loss.item()}

    return cross_entropy


def prepare_dataset(cfg: Config) -> None:

    name = str(cfg.dataset.name)
    if name == 'sort_of_clevr':
        from ..tasks.sort_of_clevr.generator import prepare_sort_of_clevr as _prepare
    elif name == 'sqoop':
        from ..tasks.sqoop.generator import prepare_sqoop as _prepare
    else:
        raise ValueError(f'unknown dataset: {name!r}')
    _prepare(cfg.dataset) # type: ignore
