#!/usr/bin/env python
"""Convert object_segmentation_real_v3_1088 splits to COCO instance JSON.

Output: data/coco_converted/{train,val,test}.json
Each output file validates cleanly with pycocotools.coco.COCO().

Usage:
    python scripts/prepare_stage1_coco.py \\
        [--dataset data/object_segmentation_real_v3_1088] \\
        [--out data/coco_converted]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kip import CLASS_NAMES
from kip.data.yolo_to_coco import convert_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=str(_ROOT / "data" / "object_segmentation_real_v3_1088"),
        help="Path to the real_v3 YOLO-seg dataset root.",
    )
    parser.add_argument(
        "--out",
        default=str(_ROOT / "data" / "coco_converted"),
        help="Output directory for COCO JSON files.",
    )
    args = parser.parse_args()

    ds_root = Path(args.dataset)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ds_root.is_dir():
        print(f"ERROR: dataset not found: {ds_root}", file=sys.stderr)
        sys.exit(1)

    # Offset IDs so train/val/test share a consistent namespace when merged
    img_id, ann_id = 1, 1

    for split in ["train", "val", "test"]:
        images_dir = ds_root / "images" / split
        labels_dir = ds_root / "labels" / split
        out_json = out_dir / f"{split}.json"

        if not images_dir.is_dir():
            print(f"  [SKIP] images dir not found: {images_dir}")
            continue

        print(f"Converting {split} ...", end=" ", flush=True)
        coco = convert_split(
            images_dir=images_dir,
            labels_dir=labels_dir,
            class_names=CLASS_NAMES,
            out_json=out_json,
            start_ids=(img_id, ann_id),
        )
        n_imgs = len(coco["images"])
        n_anns = len(coco["annotations"])
        print(f"{n_imgs} images, {n_anns} annotations -> {out_json}")

        # Advance ID counters so the next split's IDs don't collide
        if coco["images"]:
            img_id = max(im["id"] for im in coco["images"]) + 1
        if coco["annotations"]:
            ann_id = max(a["id"] for a in coco["annotations"]) + 1

    # --- Validate with pycocotools ---
    print("\nValidating with pycocotools.coco.COCO ...")
    from pycocotools.coco import COCO

    for split in ["train", "val", "test"]:
        json_path = out_dir / f"{split}.json"
        if not json_path.exists():
            print(f"  [SKIP] {json_path} not found")
            continue
        try:
            coco_api = COCO(str(json_path))
            n_imgs = len(coco_api.imgs)
            n_anns = len(coco_api.anns)
            print(f"  {split}: OK  ({n_imgs} images, {n_anns} annotations)")
        except Exception as e:
            print(f"  {split}: FAIL  {e}", file=sys.stderr)
            sys.exit(1)

    print("\nDone. COCO JSONs written to:", out_dir)


if __name__ == "__main__":
    main()
