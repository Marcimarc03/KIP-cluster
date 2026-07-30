"""PatchCore: ResNet-18 memory-bank anomaly detection (unsupervised).

Methodology: embedding / nearest-neighbour in a coreset memory bank of
defect-free patch features. Anomaly = distance to the nearest normal patch.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from kip.stage2.base import AnomalyMethod, ResNetFeatures


class PatchCore(AnomalyMethod):
    name = "patchcore"

    def __init__(self, cfg, device: str = "cpu"):
        super().__init__(cfg, device)
        self.extractor = ResNetFeatures(device, layers=("layer2", "layer3"))
        self.memory: torch.Tensor | None = None

    def _embed(self, tile):
        f, (gh, gw) = self.extractor(tile)          # (C, gh, gw)
        p = f.permute(1, 2, 0).reshape(-1, f.shape[0])
        return F.normalize(p, dim=1), (gh, gw)

    def fit(self, good_tiles) -> None:
        banks = [self._embed(t)[0] for t in good_tiles]
        bank = torch.cat(banks, dim=0)
        # coreset subsample only for large banks (Risk-7)
        ratio = getattr(self.cfg, "coreset_ratio", 0.1)
        if bank.shape[0] > 10000 and 0 < ratio < 1:
            idx = torch.randperm(bank.shape[0])[: max(1, int(bank.shape[0] * ratio))]
            bank = bank[idx]
        self.memory = bank.to(self.device)           # keep memory bank on GPU

    @torch.no_grad()
    def score(self, tile) -> tuple[np.ndarray, float]:
        emb, (gh, gw) = self._embed(tile)
        d = torch.cdist(emb.to(self.device), self.memory)   # nearest-neighbour on GPU
        patch = d.min(dim=1).values.reshape(gh, gw).cpu()
        amap = F.interpolate(patch[None, None].float(), size=tile.shape[:2],
                             mode="bilinear", align_corners=False)[0, 0].numpy()
        q = getattr(self.cfg, "fg_quantile", 0.98)
        return amap.astype(np.float32), float(np.quantile(amap, q))
