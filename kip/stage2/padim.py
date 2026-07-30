"""PaDiM: per-position multivariate-Gaussian anomaly detection (unsupervised).

Methodology: distribution modelling. Each feature-grid position gets a Gaussian
N(mu, Sigma) fitted on defect-free tiles; anomaly = Mahalanobis distance.
Complementary to PatchCore (parametric density vs. non-parametric memory bank).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from kip.stage2.base import AnomalyMethod, ResNetFeatures


class PaDiM(AnomalyMethod):
    name = "padim"

    def __init__(self, cfg, device: str = "cpu"):
        super().__init__(cfg, device)
        # layer2 only keeps the covariance tractable on tiny data
        self.extractor = ResNetFeatures(device, layers=("layer2",))
        self.rng = np.random.default_rng(getattr(cfg, "seed", 42))
        self.dim_idx = None
        self.mean = None
        self.inv_cov = None
        self.grid = None

    def _feat(self, tile):
        f, (gh, gw) = self.extractor(tile)           # (C, gh, gw)
        return f.reshape(f.shape[0], -1).numpy(), (gh, gw)   # (C, P)

    def fit(self, good_tiles) -> None:
        feats, grid = [], None
        for t in good_tiles:
            f, grid = self._feat(t)
            feats.append(f)
        self.grid = grid
        X = np.stack(feats, axis=0)                   # (N, C, P)
        n, c, p = X.shape
        d = min(getattr(self.cfg, "d_reduced", 100), c)
        self.dim_idx = self.rng.choice(c, size=d, replace=False)
        X = X[:, self.dim_idx, :]                     # (N, d, P)
        self.mean = X.mean(axis=0)                    # (d, P)
        eps = 0.01
        self.inv_cov = np.empty((p, d, d), dtype=np.float64)
        ident = np.eye(d)
        for pos in range(p):
            Xp = X[:, :, pos] - self.mean[:, pos]     # (N, d)
            cov = (Xp.T @ Xp) / max(1, n - 1) + eps * ident   # shrinkage (Risk-7)
            self.inv_cov[pos] = np.linalg.inv(cov)

    def score(self, tile) -> tuple[np.ndarray, float]:
        f, (gh, gw) = self._feat(tile)
        X = f[self.dim_idx, :]                         # (d, P)
        p = X.shape[1]
        dist = np.empty(p)
        for pos in range(p):
            delta = X[:, pos] - self.mean[:, pos]
            dist[pos] = np.sqrt(max(0.0, delta @ self.inv_cov[pos] @ delta))
        patch = torch.from_numpy(dist.reshape(gh, gw)).float()
        amap = F.interpolate(patch[None, None], size=tile.shape[:2],
                             mode="bilinear", align_corners=False)[0, 0].numpy()
        q = getattr(self.cfg, "fg_quantile", 0.98)
        return amap.astype(np.float32), float(np.quantile(amap, q))
