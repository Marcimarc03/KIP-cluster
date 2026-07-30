"""BGAD defect-dataset manifest: build, validate, save (BUILD_PLAN 2.1).

Handles the documented BGAD quirks:
- Identical mask files duplicated across train/masks and val/masks -> dedup
  by filename, keeping the first occurrence (train before val).
- Cross-split masks: a mask may physically live in the other split's masks
  dir; pairing is done via the image stem, so the mask is attached to the
  image wherever that image lives.
- ``_v2`` stem token is part of the image stem
  (``tool10_..._0008_isolated_v2.jpg`` <-> ``..._isolated_v2_polishing_wear.png``).
- Defect types are metadata only (no multi-class claims).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pandas as pd

MASK_STEM_RE = re.compile(r"(.*_isolated(?:_v2)?)_(.+)\.png$")

_SPLITS = ("train", "val")
_IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


class ManifestError(Exception):
    """Fatal manifest problem (missing files, duplicates, policy violation)."""


def normalize_defect_type(raw: str) -> tuple[str, bool]:
    """'v2_polishing_wear' -> ('polishing_wear', True); plain types pass through."""
    if raw.startswith("v2_"):
        return raw[len("v2_"):], True
    return raw, False


def _collect_masks(bgad_root: Path) -> dict[str, list[tuple[Path, str]]]:
    """Map image stem -> [(mask_path, defect_type)], deduped by mask filename.

    Masks with identical filenames in several split dirs are counted once;
    the first occurrence wins (train before val).
    """
    seen: set[str] = set()
    by_stem: dict[str, list[tuple[Path, str]]] = {}
    for split in _SPLITS:
        mask_dir = bgad_root / split / "masks"
        if not mask_dir.is_dir():
            continue
        for p in sorted(mask_dir.iterdir()):
            if p.suffix.lower() != ".png":
                continue
            if p.name in seen:
                continue  # dedup across split dirs
            seen.add(p.name)
            m = MASK_STEM_RE.match(p.name)
            if not m:
                continue  # reported by validate_manifest
            stem, raw_type = m.group(1), m.group(2)
            dtype, _is_v2 = normalize_defect_type(raw_type)
            by_stem.setdefault(stem, []).append((p, dtype))
    return by_stem


def build_manifest(
    bgad_root,
    missing_mask_policy: Literal["normal", "unlabeled", "error"] = "normal",
) -> pd.DataFrame:
    """Build the BGAD manifest DataFrame.

    Columns: image, tool_id, split, defect_status(good|defect|unlabeled),
    defect_types(';'-joined), mask_paths(';'-joined), width, height.
    """
    bgad_root = Path(bgad_root)
    masks_by_stem = _collect_masks(bgad_root)

    rows = []
    for split in _SPLITS:
        img_dir = bgad_root / split / "base_images"
        if not img_dir.is_dir():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in _IMG_SUFFIXES:
                continue
            stem = img_path.stem
            tool_id = stem.split("_")[0]
            paired = sorted(masks_by_stem.get(stem, []), key=lambda t: t[0].name)

            if paired:
                status = "defect"
            elif missing_mask_policy == "normal":
                status = "good"
            elif missing_mask_policy == "unlabeled":
                status = "unlabeled"
            else:
                raise ManifestError(
                    f"missing_mask_policy='error': image without mask: {img_path}"
                )

            img = cv2.imread(str(img_path))
            if img is None:
                raise ManifestError(f"Unreadable image: {img_path}")
            h, w = img.shape[:2]

            rows.append({
                "image": str(img_path),
                "tool_id": tool_id,
                "split": split,
                "defect_status": status,
                "defect_types": ";".join(d for _, d in paired),
                "mask_paths": ";".join(str(p) for p, _ in paired),
                "width": w,
                "height": h,
            })

    df = pd.DataFrame(rows, columns=[
        "image", "tool_id", "split", "defect_status",
        "defect_types", "mask_paths", "width", "height",
    ])
    df = df.sort_values(["split", "image"]).reset_index(drop=True)
    if df["image"].duplicated().any():
        raise ManifestError("Duplicate image entries in manifest.")
    return df


def union_masks(mask_paths, size_hw) -> np.ndarray:
    """Union of binary masks as uint8 {0,1} of shape size_hw."""
    h, w = int(size_hw[0]), int(size_hw[1])
    out = np.zeros((h, w), dtype=np.uint8)
    for p in mask_paths:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise ManifestError(f"Unreadable mask: {p}")
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        out |= (m > 127).astype(np.uint8)
    return out


def validate_manifest(df: pd.DataFrame, bgad_root) -> list[str]:
    """Return a list of warnings; raise ManifestError on fatal problems."""
    bgad_root = Path(bgad_root)
    warnings: list[str] = []

    if df["image"].duplicated().any():
        raise ManifestError("Duplicate image entries in manifest.")

    for _, row in df.iterrows():
        img_path = Path(row["image"])
        if not img_path.exists():
            raise ManifestError(f"Image missing on disk: {img_path}")
        if not re.match(r"^tool\d+$", str(row["tool_id"])):
            warnings.append(f"Unusual tool_id '{row['tool_id']}' for {img_path.name}")

        mask_str = row["mask_paths"] if isinstance(row["mask_paths"], str) else ""
        for mp in filter(None, mask_str.split(";")):
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise ManifestError(f"Unreadable mask: {mp}")
            values = np.unique(m)
            if not set(values.tolist()).issubset({0, 255}):
                warnings.append(f"Non-binary mask values in {mp}: {values[:10].tolist()}")
            if m.max() == 0:
                warnings.append(f"Empty (all-zero) mask: {mp}")
            if (m.shape[0], m.shape[1]) != (row["height"], row["width"]):
                warnings.append(
                    f"Mask size {m.shape} != image size "
                    f"({row['height']}, {row['width']}) for {mp}"
                )

    # Orphan masks: mask files whose stem matches no image in the manifest
    stems = {Path(p).stem for p in df["image"]}
    for split in _SPLITS:
        mask_dir = bgad_root / split / "masks"
        if not mask_dir.is_dir():
            continue
        for p in sorted(mask_dir.glob("*.png")):
            m = MASK_STEM_RE.match(p.name)
            if not m:
                warnings.append(f"Mask filename does not match stem regex: {p.name}")
            elif m.group(1) not in stems:
                warnings.append(f"Orphan mask (no matching image): {p.name}")
    return warnings


def save_manifest(df: pd.DataFrame, out_dir) -> Path:
    """Write manifest.csv + manifest_meta.json; return the CSV path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "manifest.csv"
    df.to_csv(csv_path, index=False, lineterminator="\n")  # LF auch unter Windows

    per_split = {}
    for split in sorted(df["split"].unique()):
        sub = df[df["split"] == split]
        per_split[split] = {
            "n_images": int(len(sub)),
            "n_good": int((sub["defect_status"] == "good").sum()),
            "n_defect": int((sub["defect_status"] == "defect").sum()),
        }
    meta = {
        "n_images": int(len(df)),
        "n_tools": int(df["tool_id"].nunique()),
        "n_good": int((df["defect_status"] == "good").sum()),
        "n_defect": int((df["defect_status"] == "defect").sum()),
        "n_unlabeled": int((df["defect_status"] == "unlabeled").sum()),
        "tools": sorted(df["tool_id"].unique().tolist()),
        "per_split": per_split,
        "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest()[:16],
    }
    (out_dir / "manifest_meta.json").write_text(json.dumps(meta, indent=2), newline="\n")
    return csv_path
