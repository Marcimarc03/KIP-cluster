"""COCO detection metric wrappers."""
from __future__ import annotations

from pycocotools.cocoeval import COCOeval


def coco_summary(cocoeval: COCOeval) -> dict:
    """Extract a flat summary dict from a COCOeval object.

    Assumes `cocoeval.evaluate()` and `cocoeval.accumulate()` have already
    been called (or use `cocoeval.summarize()` implicitly).

    Returns
    -------
    Dict with keys:
        ap50_95, ap50, ap75, ap_s, ap_m, ap_l,
        ar_det1, ar_det10, ar_det100, ar_s, ar_m, ar_l
    """
    try:
        stats = cocoeval.stats
    except AttributeError:
        cocoeval.evaluate()
        cocoeval.accumulate()
        cocoeval.summarize()
        stats = cocoeval.stats

    keys = [
        "ap50_95", "ap50", "ap75",
        "ap_s", "ap_m", "ap_l",
        "ar_det1", "ar_det10", "ar_det100",
        "ar_s", "ar_m", "ar_l",
    ]
    return {k: float(v) for k, v in zip(keys, stats)}
