"""YOLO-seg label conversion to COCO instance JSON (BUILD_PLAN 2.4).

Polygons are denormalised to absolute pixels, bbox is taken from the polygon
extent, area is computed by rasterising the polygon with cv2. Images with no
or empty label files get zero annotations; polygons with fewer than 3 points
are skipped.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

_IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_yolo_seg_line(line: str, w: int, h: int) -> tuple[int, np.ndarray]:
    """Parse one YOLO-seg label line -> (class_id, Nx2 absolute-pixel points).

    Raises ValueError for malformed lines: odd coordinate count or fewer than
    3 polygon points.
    """
    tokens = line.split()
    if len(tokens) < 2:
        raise ValueError(f"Malformed YOLO-seg line (too few tokens): {line!r}")
    cls = int(tokens[0])
    coords = [float(t) for t in tokens[1:]]
    if len(coords) % 2 != 0:
        raise ValueError(f"Odd number of polygon coordinates: {line!r}")
    n_pts = len(coords) // 2
    if n_pts < 3:
        raise ValueError(f"Polygon has fewer than 3 points ({n_pts}): {line!r}")
    pts = np.array(coords, dtype=np.float64).reshape(-1, 2)
    pts[:, 0] *= w
    pts[:, 1] *= h
    return cls, pts


def convert_split(
    images_dir,
    labels_dir,
    class_names,
    out_json,
    start_ids: tuple[int, int] = (1, 1),
) -> dict:
    """Convert one YOLO-seg split to a COCO instance-segmentation dict + JSON file."""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    out_json = Path(out_json)

    categories = [
        {"id": i + 1, "name": name, "supercategory": "component"}
        for i, name in enumerate(class_names)
    ]

    images, annotations = [], []
    img_id, ann_id = int(start_ids[0]), int(start_ids[1])

    for img_path in sorted(p for p in images_dir.iterdir()
                           if p.suffix.lower() in _IMG_SUFFIXES):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h,
        })

        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    cls, pts = parse_yolo_seg_line(line, w=w, h=h)
                except ValueError:
                    continue  # skip degenerate polygons (<3 points, odd coords)
                if not (0 <= cls < len(class_names)):
                    continue

                # Area via rasterisation
                raster = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(raster, [np.round(pts).astype(np.int32)], 1)
                area = float(raster.sum())
                if area <= 0:
                    continue

                xs, ys = pts[:, 0], pts[:, 1]
                x0, y0 = float(xs.min()), float(ys.min())
                bbox = [x0, y0, float(xs.max() - x0), float(ys.max() - y0)]

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls + 1,
                    "segmentation": [pts.flatten().tolist()],
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                })
                ann_id += 1
        img_id += 1

    coco = {"images": images, "annotations": annotations, "categories": categories}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(coco))
    return coco
