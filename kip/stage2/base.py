"""Stage-2 method base classes, shared feature extractor, registry.

Every unsupervised method implements the same contract:
    fit(good_tiles: list[BGR ndarray]) -> None
    score(tile: BGR ndarray) -> (anomaly_map HxW float32, image_score float)

The supervised U-Net (needs masks) exposes the same `score()` signature but is
fitted via `fit_supervised()` and is therefore not an AnomalyMethod subclass.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as tvm
from torchvision.transforms import functional as TF

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class ResNetFeatures:
    """Shared ResNet-18 layer2(+layer3) patch-feature extractor (frozen)."""

    def __init__(self, device: str = "cpu", layers=("layer2", "layer3")):
        self.device = device
        self.layers = layers
        net = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT).eval().to(device)
        for p in net.parameters():
            p.requires_grad_(False)
        self._feat: dict = {}
        if "layer2" in layers:
            net.layer2.register_forward_hook(
                lambda m, i, o: self._feat.__setitem__("layer2", o))
        if "layer3" in layers:
            net.layer3.register_forward_hook(
                lambda m, i, o: self._feat.__setitem__("layer3", o))
        self.net = net

    @torch.no_grad()
    def __call__(self, tile_bgr: np.ndarray, size: int = 224):
        """Return (feature_map C x gh x gw tensor on CPU, (gh, gw))."""
        rgb = cv2.cvtColor(cv2.resize(tile_bgr, (size, size)), cv2.COLOR_BGR2RGB)
        t = TF.normalize(TF.to_tensor(rgb), _IMAGENET_MEAN, _IMAGENET_STD)
        self.net(t.unsqueeze(0).to(self.device))
        maps = [self._feat[layer] for layer in self.layers]
        gh, gw = maps[0].shape[-2:]
        maps = [F.interpolate(m, size=(gh, gw), mode="bilinear", align_corners=False)
                if m.shape[-2:] != (gh, gw) else m for m in maps]
        f = torch.cat(maps, dim=1).squeeze(0)          # (C, gh, gw)
        return f.cpu(), (gh, gw)


class AnomalyMethod(ABC):
    """Unsupervised anomaly-detection contract."""

    name: str = "base"

    def __init__(self, cfg, device: str = "cpu"):
        self.cfg = cfg
        self.device = device

    @abstractmethod
    def fit(self, good_tiles) -> None: ...

    @abstractmethod
    def score(self, tile: np.ndarray) -> tuple[np.ndarray, float]: ...


def normalize_fold_scores(scores, good_ref_scores) -> np.ndarray:
    """Robust per-fold normalization: (s - median(good)) / (IQR(good) + eps).

    Required before pooling image scores across LOTO folds (AUROC is rank-based
    but pooling raw cross-fold magnitudes is biased when good-baselines differ).
    """
    scores = np.asarray(scores, dtype=float)
    good = np.asarray(good_ref_scores, dtype=float)
    if good.size == 0:
        return scores
    med = np.median(good)
    iqr = np.subtract(*np.percentile(good, [75, 25]))
    return (scores - med) / (abs(iqr) + 1e-9)


_REGISTRY: dict = {}


def _register(key, cls):
    _REGISTRY[key] = cls


def build_method(method: str, cfg, device: str):
    """Factory. 'unet' is handled by the runner (supervised), not here."""
    from kip.stage2.autoencoder import ConvAE
    from kip.stage2.padim import PaDiM
    from kip.stage2.patchcore import PatchCore

    table = {"patchcore": PatchCore, "padim": PaDiM, "ae": ConvAE}
    if method not in table:
        raise ValueError(f"Unknown unsupervised method: {method}")
    return table[method](cfg, device)
