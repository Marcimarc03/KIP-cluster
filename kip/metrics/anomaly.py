"""Anomaly detection metrics with degenerate-case guards.

All functions return None rather than raising when there is insufficient
data (e.g. single-class label set, empty masks).

Conventions
-----------
- image_auroc: labels in {0,1}, scores any real. Returns None if single class.
- pixel_auroc: gt binary 2-D, amap float32 2-D same shape. Returns None if no variation.
- aupro: per-region AUROC averaged up to fpr_limit. Returns None if no defect regions.
- best_f1: threshold sweep on binary labels. Returns (f1, threshold).
- dice_iou: empty-vs-empty -> 1.0 (MVTec convention).
             empty-GT nonempty-pred -> 0.0.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, f1_score


# ---------------------------------------------------------------------------
# Image-level metrics
# ---------------------------------------------------------------------------

def image_auroc(
    labels: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
) -> float | None:
    """Image-level AUROC. Returns None if labels contain only one class."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        return None
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return None


def best_f1(
    labels: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
) -> tuple[float, float]:
    """Find threshold maximising F1 on image-level (binary) labels.

    Returns (best_f1_score, threshold).
    Falls back to (0.0, 0.0) when no valid threshold exists.
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        return 0.0, float(scores.mean()) if len(scores) else 0.0

    prec, rec, thresholds = precision_recall_curve(labels, scores)
    # prec/rec have len = n+1; thresholds has len = n
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    if len(f1s) == 0:
        return 0.0, 0.0
    best_idx = int(np.argmax(f1s))
    return float(f1s[best_idx]), float(thresholds[best_idx])


# ---------------------------------------------------------------------------
# Pixel-level metrics
# ---------------------------------------------------------------------------

def pixel_auroc(
    gt: np.ndarray,
    amap: np.ndarray,
) -> float | None:
    """Pixel-level AUROC. Returns None if gt has only one unique value."""
    gt_flat = gt.ravel().astype(int)
    amap_flat = amap.ravel().astype(float)
    if len(np.unique(gt_flat)) < 2:
        return None
    try:
        return float(roc_auc_score(gt_flat, amap_flat))
    except Exception:
        return None


def aupro(
    gt_masks: list[np.ndarray],
    amaps: list[np.ndarray],
    fpr_limit: float = 0.3,
) -> float | None:
    """Per-Region Overlap (PRO) AUC, integrated up to fpr_limit.

    Regions are connected components (scipy.ndimage.label) of each binary GT mask.
    Micro-averages: pool all regions across images then compute the curve.

    Returns None when no defect regions exist.
    """
    from scipy.ndimage import label as ndlabel
    from scipy.interpolate import interp1d

    region_scores: list[float] = []
    region_exists: list[bool] = []
    neg_scores: list[float] = []

    for gt, amap in zip(gt_masks, amaps):
        gt_bin = (gt > 0).astype(np.uint8)

        # Negative pixels for FPR estimation
        neg_mask = gt_bin == 0
        if neg_mask.any():
            neg_scores.extend(amap[neg_mask].tolist())

        # Positive regions
        labeled, n_regions = ndlabel(gt_bin)
        if n_regions == 0:
            continue
        for region_id in range(1, n_regions + 1):
            region_mask = labeled == region_id
            region_amap = amap[region_mask]
            region_scores.append(float(region_amap.max()))
            region_exists.append(True)

    if not region_scores:
        return None  # no defect regions

    if not neg_scores:
        return None

    # Build PRO curve: for each threshold, compute FPR (global neg) and per-region TPR
    neg_arr = np.array(neg_scores, dtype=float)
    reg_arr = np.array(region_scores, dtype=float)

    all_scores = np.concatenate([neg_arr, reg_arr])
    thresholds = np.unique(all_scores)

    # Vectorised PRO curve: identical to computing (neg_arr >= thr).mean() and
    # (reg_arr >= thr).mean() for every threshold, but via searchsorted on the
    # sorted arrays instead of an O(n_thresholds * n_pixels) Python loop.
    neg_sorted = np.sort(neg_arr)
    reg_sorted = np.sort(reg_arr)
    n_neg = neg_sorted.size
    n_reg = reg_sorted.size
    fprs_arr = (n_neg - np.searchsorted(neg_sorted, thresholds, side="left")) / n_neg
    pros_arr = (n_reg - np.searchsorted(reg_sorted, thresholds, side="left")) / n_reg

    # Sort by FPR
    order = np.argsort(fprs_arr)
    fprs_arr = fprs_arr[order]
    pros_arr = pros_arr[order]

    # Clip to fpr_limit and integrate (trapz normalised)
    mask = fprs_arr <= fpr_limit
    if not mask.any():
        return None

    f = fprs_arr[mask]
    p = pros_arr[mask]

    # Add boundary point at fpr_limit using interpolation
    if f[-1] < fpr_limit and len(f) > 1:
        try:
            interp = interp1d(fprs_arr, pros_arr, bounds_error=False, fill_value=(pros_arr[0], pros_arr[-1]))
            p_boundary = float(interp(fpr_limit))
            f = np.append(f, fpr_limit)
            p = np.append(p, p_boundary)
        except Exception:
            pass

    if len(f) < 2:
        return float(p[0]) if len(p) > 0 else None

    # np.trapezoid added in NumPy 2.0 (np.trapz deprecated/removed)
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    auc_val = float(_trapz(p, f) / fpr_limit)
    return auc_val


# ---------------------------------------------------------------------------
# Pixel segmentation quality
# ---------------------------------------------------------------------------

def dice_iou(
    gt: np.ndarray,
    pred_bin: np.ndarray,
) -> dict:
    """Compute Dice and IoU for binary masks.

    Degenerate cases (MVTec convention):
    - Empty GT + empty pred  -> Dice=1.0, IoU=1.0  (trivially correct)
    - Empty GT + nonempty pred -> Dice=0.0, IoU=0.0 (false positives only)
    - Nonempty GT + empty pred -> Dice=0.0, IoU=0.0 (false negatives only)
    """
    gt = (np.asarray(gt) > 0).astype(bool)
    pred = (np.asarray(pred_bin) > 0).astype(bool)

    tp = float((gt & pred).sum())
    fp = float((~gt & pred).sum())
    fn = float((gt & ~pred).sum())

    gt_empty = gt.sum() == 0
    pred_empty = pred.sum() == 0

    if gt_empty and pred_empty:
        return {"dice": 1.0, "iou": 1.0}

    if gt_empty or pred_empty:
        return {"dice": 0.0, "iou": 0.0}

    dice = 2 * tp / (2 * tp + fp + fn + 1e-9)
    iou = tp / (tp + fp + fn + 1e-9)
    return {"dice": float(dice), "iou": float(iou)}
