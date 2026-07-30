"""Stage-2 defect-benchmark runner.

Pipeline: manifest -> Stage-1 spindle crop -> tiling -> per-fold fit/score ->
stitch -> per-fold + pooled metrics -> schema-conformant run dir.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from kip import __version__
from kip.config import Stage2Config
from kip.data.bgad_manifest import union_masks
from kip.data.splits import (assert_no_tool_leakage, fixed_split, group_kfold,
                             leave_one_tool_out)
from kip.data.tiling import Tile, stitch, tile_image
from kip.hardware import hardware_info
from kip.metrics.anomaly import aupro, best_f1, dice_iou, image_auroc, pixel_auroc
from kip.reporting.results_io import append_summary, create_run_dir, save_run
from kip.stage2.base import build_method, normalize_fold_scores

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CKPT = REPO / "results/results/yolo_runs/C_synth_pretrain_real_finetune/weights/best.pt"
DEFAULT_MANIFEST = REPO / "results/defect_detection/manifest/manifest.csv"
OUT_BASE = REPO / "results/defect_detection"


class _TCfg:
    """Minimal TilingConfig-compatible object for the crop-tiling step."""
    def __init__(self, tile_size, overlap=0.25, white_thresh=240,
                 white_frac=0.98, min_fg_coverage=0.99):
        self.tile_size = tile_size
        self.overlap = overlap
        self.white_thresh = white_thresh
        self.white_frac = white_frac
        self.min_fg_coverage = min_fg_coverage


def _prepare(manifest: pd.DataFrame, ckpt: Path, work: int, device: str, imgsz: int):
    """Return list of per-image dicts: crop (work x work BGR), gt (work), label, tool, split."""
    from ultralytics import YOLO

    from kip.stage1.crop import crop_spindle
    yolo = YOLO(str(ckpt))
    items, crop_failures = [], 0
    for _, row in manifest.iterrows():
        img = cv2.imread(str(REPO / row["image"]))
        h, w = img.shape[:2]
        out = crop_spindle(img, yolo, imgsz=imgsz, device=device, fallback="best-box")
        # crop_failures = images where the spindle class itself was not found
        res = out[1] if out else (0, 0, w, h)
        x1, y1, x2, y2 = res
        crop = img[y1:y2, x1:x2]
        # GT mask
        if isinstance(row["mask_paths"], str) and row["mask_paths"].strip():
            paths = [REPO / p for p in row["mask_paths"].split(";") if p.strip()]
            full = union_masks(paths, (h, w))
        else:
            full = np.zeros((h, w), np.uint8)
        gt_crop = full[y1:y2, x1:x2]
        crop_r = cv2.resize(crop, (work, work))
        gt_r = cv2.resize(gt_crop.astype(np.uint8), (work, work),
                          interpolation=cv2.INTER_NEAREST)
        items.append({"crop": crop_r, "gt": (gt_r > 0).astype(np.uint8),
                      "label": int(row["defect_status"] == "defect"),
                      "image": row["image"],
                      "defect_types": row["defect_types"] if isinstance(row["defect_types"], str) else "",
                      "tool": row["tool_id"], "split": row["split"]})
    return items, crop_failures


def _tiles_of(crop, gt, tcfg):
    """Return list of (Tile, tile_bgr, tile_mask)."""
    out = []
    for i, (t, tb, tm) in enumerate(tile_image(crop, gt, tcfg)):
        t.image_id = str(i)
        out.append((t, tb, tm if tm is not None else np.zeros(tb.shape[:2], np.uint8)))
    return out


def _score_image(method, crop, tcfg, work, fg_q):
    """Tile -> score -> stitch to a work-sized anomaly map + image score."""
    amaps = []
    for t, tb, _ in _tiles_of(crop, None, tcfg):
        amap, _ = method.score(tb)
        amaps.append((t, amap))
    if not amaps:
        return np.zeros((work, work), np.float32), 0.0
    full = stitch(amaps, (work, work), mode="max", fill_value=0.0)
    return full, float(np.quantile(full, fg_q))


def run_stage2(cfg: Stage2Config, ckpt: Path = DEFAULT_CKPT,
               manifest_path: Path = DEFAULT_MANIFEST, imgsz: int = 1024) -> Path:
    from kip.config import seed_everything
    seed_everything(cfg.seed)
    manifest = pd.read_csv(manifest_path)
    work = cfg.tile_size * 2
    tcfg = _TCfg(cfg.tile_size)
    items, crop_failures = _prepare(manifest, ckpt, work, cfg.device, imgsz)

    if cfg.split == "loto":
        folds = leave_one_tool_out(manifest)
    elif cfg.split == "gkf":
        folds = group_kfold(manifest, n_splits=min(5, manifest["tool_id"].nunique()), seed=cfg.seed)
    else:
        folds = [fixed_split(manifest)]

    per_fold, pooled_labels, pooled_scores = [], [], []
    pix_gts, pix_amaps = [], []
    records = []
    for fold in folds:
        if cfg.split != "fixed":
            assert_no_tool_leakage(fold, manifest)   # skipped only for fixed (Risk/WP1 note)
        tr = [items[i] for i in fold.train_idx]
        te = [items[i] for i in fold.test_idx]
        good_tr = [it for it in tr if it["label"] == 0]
        if cfg.method == "unet":
            from kip.stage2.unet import UNetSupervised
            method = UNetSupervised(cfg, cfg.device)
            tr_tiles = [tb for it in tr for _, tb, _ in _tiles_of(it["crop"], it["gt"], tcfg)]
            tr_masks = [tm for it in tr for _, _, tm in _tiles_of(it["crop"], it["gt"], tcfg)]
            if not any(m.sum() > 0 for m in tr_masks):
                per_fold.append({"fold": fold.name, "held_out_tools": fold.held_out_tools,
                                 "n_test": len(te), "skipped": "no positive train pixels"})
                continue
            method.fit_supervised(tr_tiles, tr_masks)
        else:
            if not good_tr:
                per_fold.append({"fold": fold.name, "held_out_tools": fold.held_out_tools,
                                 "n_test": len(te), "skipped": "no good train images"})
                continue
            method = build_method(cfg.method, cfg, cfg.device)
            good_tiles = [tb for it in good_tr for _, tb, _ in _tiles_of(it["crop"], it["gt"], tcfg)]
            method.fit(good_tiles)

        good_ref = [_score_image(method, it["crop"], tcfg, work, cfg.fg_quantile)[1]
                    for it in good_tr]
        labels, scores = [], []
        fold_recs, fold_gts, fold_amaps = [], [], []
        for it in te:
            amap, s = _score_image(method, it["crop"], tcfg, work, cfg.fg_quantile)
            labels.append(it["label"]); scores.append(s)
            pix_gts.append(it["gt"]); pix_amaps.append(amap)
            fold_gts.append(it["gt"]); fold_amaps.append(amap)
            fold_recs.append({"image": it["image"], "tool": it["tool"],
                              "label": it["label"], "defect_types": it["defect_types"],
                              "raw_score": s, "fold": fold.name})
        norm = normalize_fold_scores(scores, good_ref)
        pooled_labels.extend(labels); pooled_scores.extend(norm.tolist())
        for rec, ns in zip(fold_recs, norm.tolist()):
            rec["norm_score"] = ns
        records.extend(fold_recs)
        fold_dp = [(g, a) for g, a in zip(fold_gts, fold_amaps) if g.max() > 0]
        per_fold.append({
            "fold": fold.name, "held_out_tools": fold.held_out_tools,
            "n_test": len(te), "n_defect": int(sum(labels)),
            "image_auroc": image_auroc(labels, scores),
            "pixel_auroc": (pixel_auroc(np.concatenate([g.ravel() for g in fold_gts]),
                                        np.concatenate([a.ravel() for a in fold_amaps]))
                            if fold_gts else None),
            "pixel_aupro": (aupro([g for g, _ in fold_dp], [a for _, a in fold_dp])
                            if fold_dp else None),
        })

    # pooled metrics
    pooled_labels = np.array(pooled_labels); pooled_scores = np.array(pooled_scores)
    img_auroc = image_auroc(pooled_labels, pooled_scores)
    f1, thr = best_f1(pooled_labels, pooled_scores)
    pix_auroc = (pixel_auroc(np.concatenate([g.ravel() for g in pix_gts]),
                             np.concatenate([a.ravel() for a in pix_amaps]))
                 if pix_gts else None)
    defect_pairs = [(g, a) for g, a in zip(pix_gts, pix_amaps) if g.max() > 0]
    pro = aupro([g for g, _ in defect_pairs], [a for _, a in defect_pairs]) if defect_pairs else None
    dice = iou = None
    if cfg.method == "unet" and defect_pairs:
        dis = [dice_iou(g, (a > 0.5).astype(np.uint8)) for g, a in defect_pairs]
        dice = float(np.mean([d["dice"] for d in dis]))
        iou = float(np.mean([d["iou"] for d in dis]))

    pooled = {"image_auroc": img_auroc, "image_f1": f1, "image_f1_threshold": thr,
              "pixel_auroc": pix_auroc, "pixel_aupro": pro, "pixel_f1": None,
              "dice": dice, "iou": iou}

    run_name = f"{cfg.method}_{cfg.split}_aug{'on' if cfg.augmentation else 'off'}_seed{cfg.seed}"
    run_dir = create_run_dir(OUT_BASE, run_name)
    metrics = {
        "run_id": run_dir.name, "stage": 2, "model": cfg.method,
        "augmentation": cfg.augmentation, "seed": cfg.seed, "smoke": cfg.smoke,
        "device": cfg.device, "kip_version": __version__,
        "split": {"scheme": cfg.split, "n_folds": len(folds),
                  "test_set": "bgad_pooled"},
        "dataset": {"n_images": len(items), "n_good": int(sum(1 for it in items if it["label"] == 0)),
                    "n_defect": int(sum(it["label"] for it in items)),
                    "missing_mask_policy": "normal", "crop_failures": crop_failures,
                    "work_size": work, "tile_size": cfg.tile_size},
        "metrics": {"pooled": pooled, "per_fold": per_fold},
    }
    save_run(run_dir, cfg, metrics, manifest_path=manifest_path, hardware=hardware_info())
    # persist per-image scores + anomaly maps for post-hoc analysis (bootstrap-CI,
    # defect-type breakdown, overlays) WITHOUT a DGX rerun
    pred_dir = run_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(pred_dir / "scores.csv", index=False)
    np.savez_compressed(
        pred_dir / "amaps.npz",
        amaps=(np.stack(pix_amaps).astype(np.float32) if pix_amaps else np.zeros((0, work, work), np.float32)),
        gts=(np.stack(pix_gts).astype(np.uint8) if pix_gts else np.zeros((0, work, work), np.uint8)),
        images=np.array([r["image"] for r in records]),
    )
    append_summary(OUT_BASE, {
        "run_id": run_dir.name, "stage": 2, "model": cfg.method,
        "augmentation": cfg.augmentation, "seed": cfg.seed, "smoke": cfg.smoke,
        "split_scheme": cfg.split, "device": cfg.device,
        "metric.image_auroc": img_auroc, "metric.pixel_auroc": pix_auroc,
        "metric.image_f1": f1, "metric.pixel_aupro": pro,
        "metric.dice": dice, "metric.iou": iou,
    })
    return run_dir
