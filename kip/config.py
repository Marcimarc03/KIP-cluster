"""Configuration dataclasses, YAML I/O, and seed_everything()."""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

@dataclass
class TilingConfig:
    tile_size: int = 576
    overlap: float = 0.25
    white_thresh: int = 240
    white_frac: float = 0.95
    min_fg_coverage: float = 0.99   # assert kept tiles cover >= this of GT pixels


@dataclass
class Stage1Config:
    model: Literal["yolo", "yolo26", "mask2former", "maskrcnn"] = "yolo"
    augmentation: bool = True
    epochs: int = 100
    imgsz: int = 1088
    batch: int = 16
    lr: float = 1e-4
    freeze_backbone_epochs: int = 0
    device: str = "cpu"
    seed: int = 42
    smoke: bool = False
    # aug-off zeros (YOLO)
    mosaic: float = 1.0
    mixup: float = 0.0
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    flipud: float = 0.0
    fliplr: float = 0.5
    scale: float = 0.5
    translate: float = 0.1
    erasing: float = 0.4


@dataclass
class Stage2Config:
    method: Literal["patchcore", "padim", "ae", "unet"] = "patchcore"
    split: Literal["loto", "gkf", "fixed"] = "loto"
    augmentation: bool = False
    epochs: int = 50          # ae/unet
    batch: int = 8
    lr: float = 1e-3
    tile_size: int = 576
    d_reduced: int = 100       # PaDiM
    coreset_ratio: float = 0.1  # PatchCore
    loss: Literal["bce_dice", "focal"] = "bce_dice"
    device: str = "cpu"
    seed: int = 42
    smoke: bool = False
    fg_quantile: float = 0.98
    crop_fallback: Literal["full", "best-box", "skip"] = "best-box"
    tiling: TilingConfig = field(default_factory=TilingConfig)


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def _cfg_to_dict(cfg) -> dict:
    d = asdict(cfg)
    # convert nested dataclass dicts already handled by asdict
    return d


def save_config(cfg, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(_cfg_to_dict(cfg), f, default_flow_style=False, sort_keys=False)


def load_stage1_config(path: Path) -> Stage1Config:
    with open(path) as f:
        d = yaml.safe_load(f)
    return Stage1Config(**{k: v for k, v in d.items() if k in Stage1Config.__dataclass_fields__})


def load_stage2_config(path: Path) -> Stage2Config:
    with open(path) as f:
        d = yaml.safe_load(f)
    tiling_d = d.pop("tiling", {})
    tiling = TilingConfig(**{k: v for k, v in tiling_d.items() if k in TilingConfig.__dataclass_fields__})
    cfg = Stage2Config(**{k: v for k, v in d.items() if k in Stage2Config.__dataclass_fields__})
    cfg.tiling = tiling
    return cfg


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    """Set all relevant seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # torch < 1.11 signature does not have warn_only
        torch.use_deterministic_algorithms(True)
