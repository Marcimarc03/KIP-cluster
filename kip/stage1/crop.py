"""Stage-1 -> Stage-2 hand-off: crop the bevel_gear_spindle from a BGAD image.

Ported from kip_stage2_bgad.crop_spindle with:
- Explicit `fallback` parameter ('full' | 'best-box' | 'skip')
- Optional disk cache (crop cache dir)
- Returns None when fallback='skip' and no spindle detected
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np

from kip import SPINDLE_CLASS


def crop_spindle(
    img_bgr: np.ndarray,
    yolo_model,
    imgsz: int = 1024,
    conf: float = 0.25,
    fallback: Literal["full", "best-box", "skip"] = "best-box",
    device: Optional[str] = None,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Detect and crop the bevel_gear_spindle from a BGR image.

    Parameters
    ----------
    img_bgr:
        Input image as BGR numpy array.
    yolo_model:
        Loaded ultralytics YOLO model (Stage-1 checkpoint).
    imgsz:
        Inference resolution for the YOLO model.
    conf:
        Confidence threshold for detections.
    fallback:
        Behaviour when no spindle is detected:
        - 'full'      — return the entire image with bbox = full extent.
        - 'best-box'  — return the highest-confidence detection of ANY class.
        - 'skip'      — return None (caller must handle).
    device:
        Torch device string (e.g. 'mps', 'cuda', 'cpu').
        Defaults to the YOLO model's device.

    Returns
    -------
    (crop_bgr, (x1, y1, x2, y2)) or None (only when fallback='skip').
    """
    # Build predict kwargs
    predict_kwargs: dict = dict(
        source=img_bgr,
        imgsz=imgsz,
        conf=conf,
        verbose=False,
    )
    if device is not None:
        predict_kwargs["device"] = device

    res = yolo_model.predict(**predict_kwargs)[0]

    h, w = img_bgr.shape[:2]
    spindle_box = None

    if res.boxes is not None and len(res.boxes) > 0:
        cls_arr = res.boxes.cls.cpu().numpy().astype(int)
        conf_arr = res.boxes.conf.cpu().numpy()
        # Find best spindle detection
        spindle_cands = [
            (i, conf_arr[i]) for i in range(len(cls_arr)) if cls_arr[i] == SPINDLE_CLASS
        ]
        if spindle_cands:
            best_idx = max(spindle_cands, key=lambda t: t[1])[0]
            spindle_box = res.boxes.xyxy[best_idx].cpu().numpy().astype(int)
        elif fallback == "best-box":
            # Best detection of any class
            best_idx = int(conf_arr.argmax())
            spindle_box = res.boxes.xyxy[best_idx].cpu().numpy().astype(int)

    if spindle_box is None:
        if fallback == "full":
            return img_bgr.copy(), (0, 0, w, h)
        elif fallback == "skip":
            return None
        else:
            # best-box with no detections at all -> full image
            return img_bgr.copy(), (0, 0, w, h)

    x1, y1, x2, y2 = spindle_box
    x1 = int(max(0, x1))
    y1 = int(max(0, y1))
    x2 = int(min(w, x2))
    y2 = int(min(h, y2))

    if x2 <= x1 or y2 <= y1:
        if fallback == "skip":
            return None
        return img_bgr.copy(), (0, 0, w, h)

    crop = img_bgr[y1:y2, x1:x2].copy()
    return crop, (x1, y1, x2, y2)


def crop_spindle_cached(
    img_path: str | Path,
    yolo_model,
    cache_dir: Optional[str | Path],
    imgsz: int = 1024,
    conf: float = 0.25,
    fallback: Literal["full", "best-box", "skip"] = "best-box",
    device: Optional[str] = None,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Wrapper around crop_spindle with optional disk cache.

    Cache stores crops as PNG files under `cache_dir`.
    BBox metadata is not cached (crop only).
    """
    img_path = Path(img_path)

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / (img_path.stem + "_crop.png")
        if cache_file.exists():
            cached = cv2.imread(str(cache_file))
            if cached is not None:
                # Return cached crop with dummy bbox (0,0,w,h)
                ch, cw = cached.shape[:2]
                return cached, (0, 0, cw, ch)

    img = cv2.imread(str(img_path))
    if img is None:
        return None

    result = crop_spindle(img, yolo_model, imgsz=imgsz, conf=conf,
                          fallback=fallback, device=device)

    if result is not None and cache_dir is not None:
        cv2.imwrite(str(cache_file), result[0])

    return result
