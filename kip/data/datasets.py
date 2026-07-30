"""Torch datasets over tiles (BUILD_PLAN WP1).

Kept intentionally small: the stage-2 runner mostly works with plain lists
of tiles; these Dataset wrappers exist for batched training (ConvAE/U-Net)
and for an optional on-disk crop cache.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    """HWC uint8 BGR -> CHW float32 RGB in [0, 1]."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0


class TileDataset(Dataset):
    """Unsupervised tile dataset (images only), e.g. for the ConvAE."""

    def __init__(self, tiles: list[np.ndarray], transform=None):
        self.tiles = tiles
        self.transform = transform

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = self.tiles[idx]
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return _to_tensor(img)


class SupervisedTileDataset(Dataset):
    """Supervised (tile, binary mask) dataset for the U-Net.

    ``transform`` must be a mask-consistent albumentations Compose
    (see kip.data.augment.stage2_augment_pipeline).
    """

    def __init__(self, tiles: list[np.ndarray], masks: list[np.ndarray], transform=None):
        assert len(tiles) == len(masks), "tiles and masks must align"
        self.tiles = tiles
        self.masks = masks
        self.transform = transform

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, mask = self.tiles[idx], self.masks[idx]
        if self.transform is not None:
            out = self.transform(image=img, mask=mask)
            img, mask = out["image"], out["mask"]
        return _to_tensor(img), torch.from_numpy((mask > 0).astype(np.float32))[None]


class CropCache:
    """Optional on-disk cache for stage-1 spindle crops keyed by image stem."""

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.png"

    def get(self, key: str):
        p = self._path(key)
        return cv2.imread(str(p)) if p.exists() else None

    def put(self, key: str, crop_bgr: np.ndarray) -> None:
        cv2.imwrite(str(self._path(key)), crop_bgr)
