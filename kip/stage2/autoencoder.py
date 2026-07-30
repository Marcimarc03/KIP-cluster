"""Convolutional autoencoder anomaly detection (unsupervised).

Methodology: reconstruction. A conv AE is trained to reconstruct defect-free
tiles; anomaly = per-pixel reconstruction error. Complementary to the
feature-embedding methods (learns to regenerate normal appearance).
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kip.stage2.base import AnomalyMethod


class _AE(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, ch, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(ch, ch * 2, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1), nn.ReLU(True),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(ch, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


class ConvAE(AnomalyMethod):
    name = "ae"

    def __init__(self, cfg, device: str = "cpu"):
        super().__init__(cfg, device)
        self.size = 128
        self.model = _AE().to(device)

    def _to_tensor(self, tile):
        rgb = cv2.cvtColor(cv2.resize(tile, (self.size, self.size)), cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0

    def fit(self, good_tiles) -> None:
        X = torch.stack([self._to_tensor(t) for t in good_tiles]).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=getattr(self.cfg, "lr", 1e-3))
        epochs = getattr(self.cfg, "epochs", 50)
        bs = max(1, min(getattr(self.cfg, "batch", 8), X.shape[0]))
        self.model.train()
        for _ in range(epochs):
            perm = torch.randperm(X.shape[0])
            for i in range(0, X.shape[0], bs):
                xb = X[perm[i:i + bs]]
                opt.zero_grad()
                loss = F.mse_loss(self.model(xb), xb)
                loss.backward()
                opt.step()
        self.model.eval()

    @torch.no_grad()
    def score(self, tile) -> tuple[np.ndarray, float]:
        x = self._to_tensor(tile).unsqueeze(0).to(self.device)
        err = ((self.model(x) - x) ** 2).mean(dim=1)[0].cpu().numpy()   # (size, size)
        err = cv2.GaussianBlur(err, (0, 0), sigmaX=4)
        amap = cv2.resize(err, (tile.shape[1], tile.shape[0]))
        q = getattr(self.cfg, "fg_quantile", 0.98)
        return amap.astype(np.float32), float(np.quantile(amap, q))
