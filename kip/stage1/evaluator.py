"""Single fair COCO evaluator for both Stage-1 models (§2.5 / §3 BUILD_PLAN).

Both YOLO and Mask2Former predictions must pass through this function so that
the paper's comparison rests on identical evaluation code.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from kip import CLASS_NAMES
from kip.metrics.detection import coco_summary


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_coco(
    gt_json: Union[str, Path],
    pred_json: Union[str, Path],
    class_names: list[str] | None = None,
) -> dict:
    """Run pycocotools bbox + segm evaluation and return §3 stage-1 metrics.

    Parameters
    ----------
    gt_json:
        Path to COCO ground-truth annotations JSON.
    pred_json:
        Path to COCO predictions JSON (list of dicts with image_id,
        category_id, bbox, segmentation, score).
    class_names:
        Ordered list of class names; defaults to kip.CLASS_NAMES.

    Returns
    -------
    dict with keys
        bbox_map50, bbox_map50_95,
        segm_map50, segm_map50_95, segm_map75,
        precision, recall,
        per_class: {<name>: {segm_ap50, segm_ap50_95}}

    Notes
    -----
    - ``precision`` = segm AR@1 detection (proxy for high-precision point)
    - ``recall``    = segm AR@100 detections (maximum achievable recall)
    - Per-class AP is -1 for categories with no GT instances.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    gt_json = Path(gt_json)
    pred_json = Path(pred_json)

    # Load ground truth
    coco_gt = COCO(str(gt_json))

    # Load predictions
    raw_preds = json.loads(pred_json.read_text())
    if not raw_preds:
        return _empty_metrics(coco_gt, class_names)

    coco_dt = coco_gt.loadRes(raw_preds)

    # ------------------------------------------------------------------
    # Bbox evaluation
    # ------------------------------------------------------------------
    eval_bbox = COCOeval(coco_gt, coco_dt, iouType="bbox")
    eval_bbox.evaluate()
    eval_bbox.accumulate()
    eval_bbox.summarize()
    bbox_s = coco_summary(eval_bbox)

    # ------------------------------------------------------------------
    # Segm evaluation (whole dataset)
    # ------------------------------------------------------------------
    eval_segm = COCOeval(coco_gt, coco_dt, iouType="segm")
    eval_segm.evaluate()
    eval_segm.accumulate()
    eval_segm.summarize()
    segm_s = coco_summary(eval_segm)

    # ------------------------------------------------------------------
    # Per-class segm AP (only for categories present in GT)
    # ------------------------------------------------------------------
    cat_ids = sorted(coco_gt.getCatIds())
    cat_id_to_name = {c["id"]: c["name"] for c in coco_gt.loadCats(cat_ids)}

    per_class: dict[str, dict] = {}
    for cat_id in cat_ids:
        cat_name = cat_id_to_name.get(cat_id, f"class_{cat_id}")
        # Check if any GT annotations exist for this category
        ann_ids = coco_gt.getAnnIds(catIds=[cat_id])
        if not ann_ids:
            per_class[cat_name] = {"segm_ap50": -1.0, "segm_ap50_95": -1.0}
            continue

        eval_c = COCOeval(coco_gt, coco_dt, iouType="segm")
        eval_c.params.catIds = [cat_id]
        eval_c.evaluate()
        eval_c.accumulate()
        eval_c.summarize()
        s = coco_summary(eval_c)
        per_class[cat_name] = {
            "segm_ap50": s["ap50"],
            "segm_ap50_95": s["ap50_95"],
        }

    return {
        "bbox_map50": bbox_s["ap50"],
        "bbox_map50_95": bbox_s["ap50_95"],
        "segm_map50": segm_s["ap50"],
        "segm_map50_95": segm_s["ap50_95"],
        "segm_map75": segm_s["ap75"],
        # precision = AR@1 detection (high-precision proxy)
        "precision": segm_s["ar_det1"],
        # recall = AR@100 detections (maximum recall)
        "recall": segm_s["ar_det100"],
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_metrics(coco_gt: COCO, class_names: list[str]) -> dict:
    """Return zero/undefined metrics when the prediction list is empty."""
    cat_ids = sorted(coco_gt.getCatIds())
    cat_id_to_name = {c["id"]: c["name"] for c in coco_gt.loadCats(cat_ids)}

    per_class: dict[str, dict] = {}
    for cat_id in cat_ids:
        cat_name = cat_id_to_name.get(cat_id, f"class_{cat_id}")
        per_class[cat_name] = {"segm_ap50": 0.0, "segm_ap50_95": 0.0}

    return {
        "bbox_map50": 0.0,
        "bbox_map50_95": 0.0,
        "segm_map50": 0.0,
        "segm_map50_95": 0.0,
        "segm_map75": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "per_class": per_class,
    }
