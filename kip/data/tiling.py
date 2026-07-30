"""Overlapping tiling with background filtering and stitching (BUILD_PLAN 2.2).

BGAD closeups are 5472x3648 with a near-white isolated background; tiles that
are almost entirely white carry no signal and are dropped. A coverage guard
ensures kept tiles still cover >= min_fg_coverage of GT-positive pixels,
otherwise the background filter is relaxed (Risk-5).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Tile:
    image_id: str
    x0: int
    y0: int
    w: int
    h: int
    fg_fraction: float


def compute_grid(h: int, w: int, tile: int = 576, overlap: float = 0.25) -> list[tuple[int, int]]:
    """Top-left corners (y0, x0) of an overlapping grid covering the image."""
    def _axis(length: int) -> list[int]:
        if length <= tile:
            return [0]
        stride = max(1, int(round(tile * (1.0 - overlap))))
        pos = list(range(0, length - tile + 1, stride))
        if pos[-1] != length - tile:
            pos.append(length - tile)
        return pos

    return [(y0, x0) for y0 in _axis(h) for x0 in _axis(w)]


def is_background_tile(tile_bgr: np.ndarray, white_thresh: int = 240, white_frac: float = 0.95) -> bool:
    """True if >= white_frac of pixels are near-white (all channels >= white_thresh)."""
    if tile_bgr.ndim == 3:
        near_white = (tile_bgr >= white_thresh).all(axis=2)
    else:
        near_white = tile_bgr >= white_thresh
    return bool(near_white.mean() >= white_frac)


def _cut(img, mask, y0, x0, tile, h, w):
    th = min(tile, h - y0)
    tw = min(tile, w - x0)
    tile_img = img[y0:y0 + th, x0:x0 + tw]
    tile_mask = mask[y0:y0 + th, x0:x0 + tw] if mask is not None else None
    return tile_img, tile_mask, th, tw


def tile_image(img: np.ndarray, mask, cfg) -> list[tuple[Tile, np.ndarray, np.ndarray | None]]:
    """Tile an image (and optional mask), dropping near-white background tiles.

    Returns list of (Tile, tile_bgr, tile_mask|None). If dropping background
    tiles would lose more than (1 - cfg.min_fg_coverage) of the GT-positive
    pixels, the background filter is progressively relaxed.
    """
    h, w = img.shape[:2]
    grid = compute_grid(h, w, tile=cfg.tile_size, overlap=cfg.overlap)

    def _build(white_thresh: int, white_frac: float):
        out = []
        for y0, x0 in grid:
            tile_img, tile_mask, th, tw = _cut(img, mask, y0, x0, cfg.tile_size, h, w)
            if white_frac <= 1.0 and is_background_tile(tile_img, white_thresh, white_frac):
                continue
            fg = float(tile_mask.mean()) if tile_mask is not None else 0.0
            out.append((Tile("", x0, y0, tw, th, fg), tile_img, tile_mask))
        return out

    tiles = _build(cfg.white_thresh, cfg.white_frac)

    # Coverage guard (Risk-5): kept tiles must cover >= min_fg_coverage of GT+.
    if mask is not None and mask.sum() > 0:
        def _coverage(tls) -> float:
            covered = np.zeros((h, w), dtype=bool)
            for t, _, _ in tls:
                covered[t.y0:t.y0 + t.h, t.x0:t.x0 + t.w] = True
            return float((mask.astype(bool) & covered).sum() / mask.astype(bool).sum())

        if _coverage(tiles) < cfg.min_fg_coverage:
            # Relax: raise threshold first, then disable filtering entirely.
            for thresh, frac in ((255, cfg.white_frac), (255, 2.0)):
                tiles = _build(thresh, frac)
                if _coverage(tiles) >= cfg.min_fg_coverage:
                    break
    return tiles


def stitch(tiles, full_hw, mode: str = "mean", fill_value: float = 0.0) -> np.ndarray:
    """Blend per-tile score maps back to full resolution.

    mode='mean': overlap-weighted average; mode='max': pixelwise maximum.
    Pixels not covered by any tile get fill_value.
    """
    h, w = int(full_hw[0]), int(full_hw[1])
    covered = np.zeros((h, w), dtype=bool)

    if mode == "mean":
        acc = np.zeros((h, w), dtype=np.float64)
        cnt = np.zeros((h, w), dtype=np.float64)
        for t, amap in tiles:
            acc[t.y0:t.y0 + t.h, t.x0:t.x0 + t.w] += amap
            cnt[t.y0:t.y0 + t.h, t.x0:t.x0 + t.w] += 1.0
            covered[t.y0:t.y0 + t.h, t.x0:t.x0 + t.w] = True
        out = np.full((h, w), fill_value, dtype=np.float32)
        np.divide(acc, cnt, out=out, where=cnt > 0, casting="unsafe")
        out[~covered] = fill_value
        return out.astype(np.float32)

    if mode == "max":
        out = np.full((h, w), -np.inf, dtype=np.float32)
        for t, amap in tiles:
            region = out[t.y0:t.y0 + t.h, t.x0:t.x0 + t.w]
            np.maximum(region, amap.astype(np.float32), out=region)
            covered[t.y0:t.y0 + t.h, t.x0:t.x0 + t.w] = True
        out[~covered] = fill_value
        return out

    raise ValueError(f"Unknown stitch mode: {mode!r}")
