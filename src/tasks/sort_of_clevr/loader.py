from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

from ..base import build_loaders, images_to_tensor, load_split, resolve_paths, to_float_images
from ...core.contracts import VQABatch

if TYPE_CHECKING:
    from ...core.config import Config

SPLITS = {'train': 'train', 'eval': 'val', 'test': 'test'}


def _load_split(path: str) -> dict:

    data = load_split(path)
    return {
        'images': data['images'],                                  
        'questions': torch.from_numpy(data['questions']).long(),
        'answers': torch.from_numpy(data['answers']).long(),
        'image_idx': torch.from_numpy(data['image_idx']).long(),
    }


class SortOfClevrDataset(Dataset):

    def __init__(self, path: str) -> None:
        super().__init__()
        self.split = _load_split(path)

    def __len__(self) -> int:
        return self.split['answers'].size(0)

    def __getitem__(self, idx: int) -> VQABatch:
        image = images_to_tensor(self.split['images'][int(self.split['image_idx'][idx])])
        return {
            'images': image,
            'questions': self.split['questions'][idx],
            'answers': self.split['answers'][idx],
        }


class SortOfClevrOnDeviceLoader:

    def __init__(
            self, path: str, batch_size: int, device: str,
            shuffle: bool = False,
            ) -> None:
        non_blocking = device.startswith('cuda')
        split = _load_split(path)
        split['images'] = images_to_tensor(split['images'])      
        self.split = {
            k: v.to(device, non_blocking=non_blocking)
            for k, v in split.items()
        }
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.N = self.split['answers'].size(0)

    def __len__(self) -> int:
        return (self.N + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[VQABatch]:
        order = (
            torch.randperm(self.N, device=self.device) if self.shuffle
            else torch.arange(self.N, device=self.device)
        )
        last = self.N - self.N % self.batch_size if self.shuffle else self.N
        for i in range(0, last, self.batch_size):
            idx = order[i: i + self.batch_size]
            images = self.split['images'][self.split['image_idx'][idx]]
            yield {
                'images': to_float_images(images),
                'questions': self.split['questions'][idx],
                'answers': self.split['answers'][idx],
            }


def build_dataloaders(cfg: Config, device: str):
    from .generator import prepare_sort_of_clevr

    paths = resolve_paths(cfg, SPLITS, prepare_sort_of_clevr)
    return build_loaders(
        cfg, device, paths, SortOfClevrDataset, SortOfClevrOnDeviceLoader)
