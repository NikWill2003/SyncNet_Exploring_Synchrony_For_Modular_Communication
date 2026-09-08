from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from ..core.config import Config


def to_float_images(images_u8: torch.Tensor) -> torch.Tensor:

    return images_u8.float().div(255.0)


def save_split(arrays: dict[str, np.ndarray], path: Path) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(arrays)
    np.save(path.with_suffix('.images.npy'), np.ascontiguousarray(arrays.pop('images')))
    np.savez_compressed(path, allow_pickle=False, **arrays)


def load_split(path: str | Path) -> dict[str, np.ndarray]:

    with np.load(path) as data:
        arrays = {k: data[k] for k in data.files}
    npy = Path(path).with_suffix('.images.npy')
    if npy.exists():
        arrays['images'] = np.load(npy, mmap_mode='r')
    return arrays


def images_to_tensor(images_u8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.array(images_u8, copy=True)).movedim(-1, -3).contiguous()   

class DeviceCastLoader:

    def __init__(self, loader: DataLoader, device: str) -> None:
        self.loader, self.device = loader, device

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        non_blocking = self.device.startswith('cuda')
        for batch in self.loader:
            batch = {k: v.to(self.device, non_blocking=non_blocking) for k, v in batch.items()}
            batch['images'] = to_float_images(batch['images'])
            yield batch


def _dataset_overrides(cfg: Config) -> str:
    keys = ('seed', 'train_size', 'test_size', 'rhs_variety', 'nb_questions', 't_subtype')
    return ' '.join(f'dataset.{k}={getattr(cfg.dataset, k)}' for k in keys if hasattr(cfg.dataset, k))


def resolve_paths(cfg: 'Config', stems: dict[str, str], prepare) -> dict[str, Path]:
    """Locate the npz for each trainer slot, generating them if missing."""
    root = Path(cfg.dataset.root) / cfg.dataset.dir
    paths = {slot: root / f'{stem}.npz' for slot, stem in stems.items()}

    if not all(path.exists() for path in paths.values()):
        if not cfg.dataset.prepare_if_missing:
            raise FileNotFoundError(
                f'dataset not found under {root}; generate it first with\n'
                f'    python -m scripts.prepare_dataset task={cfg.dataset.name} {_dataset_overrides(cfg)}\n'
                f'or set dataset.prepare_if_missing=true for a single run')
        prepare(cfg.dataset)

    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f'prepare() did not produce {missing}')

    return paths


def build_loaders(
        cfg: Config,
        device: str,
        paths: dict[str, Path],
        dataset_cls: Callable[[str], Any],
        on_device_cls: Callable[..., Any],
        ) -> tuple[Any, ...]:

    mode = cfg.train.loader_mode
    sizes = {'train': cfg.train.train_bs, 'eval': cfg.train.val_bs,
             'test': cfg.train.val_bs}
    order = ('train', 'eval', 'test')

    if mode == 'gpu_cached':
        return tuple(
            on_device_cls(
                str(paths[split]), sizes[split], device,
                shuffle=(split == 'train'),
            )
            for split in order
        )

    if mode == 'dataloader':
        return tuple(
            DeviceCastLoader(DataLoader(
                dataset_cls(str(paths[split])),
                batch_size=sizes[split], 
                shuffle=(split == 'train'),
                drop_last=(split == 'train'),                     
                num_workers=cfg.train.num_workers if split == 'train' else 0,
                persistent_workers=(split == 'train' and cfg.train.num_workers > 0),
                pin_memory=(device != 'cpu'),
            ), device)
            for split in order
        )

    raise ValueError(f'unknown loader_mode: {mode!r}')

