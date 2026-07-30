"""Augmentation specs for both stages (BUILD_PLAN 2.5 / WP1).

Stage 1 (YOLO): hyperparameter dicts passed straight into ultralytics
``model.train(**hyp)``. aug-off zeroes mosaic/mixup/hsv/flips/scale/
translate/erasing so the comparison against Mask2Former is fair.

Stage 2: mask-consistent albumentations pipeline for supervised tile
training (U-Net). Geometric + photometric transforms are applied jointly
to image and mask.
"""
from __future__ import annotations


def yolo_aug_hyp(aug_on: bool) -> dict:
    """Ultralytics augmentation hyperparameters for aug on/off."""
    if aug_on:
        return {
            "mosaic": 1.0,
            "mixup": 0.0,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "flipud": 0.0,
            "fliplr": 0.5,
            "scale": 0.5,
            "translate": 0.1,
            "erasing": 0.4,
        }
    return {
        "mosaic": 0.0,
        "mixup": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "scale": 0.0,
        "translate": 0.0,
        "erasing": 0.0,
    }


def stage2_augment_pipeline(aug_on: bool, tile_size: int | None = None):
    """Mask-consistent albumentations pipeline for stage-2 tile training.

    Returns an ``albumentations.Compose`` (or None when aug_on is False).
    Import is lazy so environments without albumentations can still use
    the rest of this module.
    """
    if not aug_on:
        return None
    import albumentations as A

    transforms = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2, p=0.5),
        A.GaussNoise(p=0.2),
    ]
    if tile_size is not None:
        transforms.append(A.PadIfNeeded(min_height=tile_size, min_width=tile_size))
    return A.Compose(transforms)
