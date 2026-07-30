"""YOLO11n-seg trainer and COCO predictor for Stage-1 benchmark.

Usage (via run_stage1.py):
    trainer = YoloSegTrainer(cfg, data_yaml, run_dir)
    ckpt_path = trainer.train()
    trainer.predict_to_coco(ckpt_path, coco_gt_json, images_dir, out_json)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Union

import cv2
import numpy as np

from kip.config import Stage1Config
from kip.data.augment import yolo_aug_hyp


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS = "yolo11n-seg.pt"
# Smoke: train on ~5% of data (~40 images from 771)
_SMOKE_FRACTION = 0.06
# AMP must be disabled on MPS to avoid metal ops not supported
_MPS_DEVICES = {"mps", "apple"}


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class YoloSegTrainer:
    """Train YOLO11n-seg and produce COCO-format predictions.

    Parameters
    ----------
    cfg:
        Stage1Config controlling epochs, imgsz, batch, device, aug, etc.
    data_yaml:
        Path to the YOLO-format data.yaml (train/val splits).
    run_dir:
        Output root for this run.  YOLO checkpoints land in
        ``<run_dir>/yolo_train/``.
    weights:
        Initial weights.  Defaults to ``yolo11n-seg.pt`` (auto-downloaded
        by ultralytics if absent).
    """

    def __init__(
        self,
        cfg: Stage1Config,
        data_yaml: Union[str, Path],
        run_dir: Path,
        weights: str = _DEFAULT_WEIGHTS,
    ) -> None:
        self.cfg = cfg
        self.data_yaml = Path(data_yaml)
        self.run_dir = Path(run_dir)
        self.weights = weights

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> Path:
        """Train YOLO11n-seg and return the path to best.pt.

        Aug off: zeros mosaic/mixup/hsv/flip/scale/translate/erasing per
        STAGE1_YOLO_HYP_OFF (fairness contract §2.5).
        """
        from ultralytics import YOLO

        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        model = YOLO(self.weights)

        hyp = yolo_aug_hyp(self.cfg.augmentation)
        is_mps = self.cfg.device.lower() in _MPS_DEVICES or self.cfg.device.startswith("mps")

        fraction = _SMOKE_FRACTION if self.cfg.smoke else 1.0

        train_kwargs: dict = {
            "data": str(self.data_yaml),
            "epochs": self.cfg.epochs,
            "imgsz": self.cfg.imgsz,
            "batch": self.cfg.batch,
            "device": self.cfg.device,
            "seed": self.cfg.seed,
            "project": str(self.run_dir),
            "name": "yolo_train",
            "exist_ok": True,
            "verbose": False,
            "amp": not is_mps,           # AMP off on MPS
            "fraction": fraction,
            # DataLoader workers: env-overridable. Set KIP_WORKERS=0 on hosts with
            # a small /dev/shm (e.g. Docker containers) to avoid shared-memory errors.
            "workers": int(os.environ.get("KIP_WORKERS", "8")),
            # Augmentation hyp dict
            **hyp,
        }

        model.train(**train_kwargs)

        # Locate best checkpoint
        ckpt_path = self.run_dir / "yolo_train" / "weights" / "best.pt"
        if not ckpt_path.exists():
            # Fallback: last.pt
            ckpt_path = self.run_dir / "yolo_train" / "weights" / "last.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Could not find YOLO checkpoint in {self.run_dir}/yolo_train/weights/. "
                "Check training output."
            )
        return ckpt_path

    # ------------------------------------------------------------------
    # Prediction -> COCO
    # ------------------------------------------------------------------

    def predict_to_coco(
        self,
        ckpt: Union[str, Path],
        coco_gt_json: Union[str, Path],
        images_dir: Union[str, Path],
        out_json: Union[str, Path],
    ) -> Path:
        """Run inference on all images listed in ``coco_gt_json`` and write
        COCO-format predictions to ``out_json``.

        Parameters
        ----------
        ckpt:
            Path to YOLO checkpoint (.pt).
        coco_gt_json:
            COCO GT JSON (used only to retrieve the ordered image list).
        images_dir:
            Directory containing the raw images.
        out_json:
            Destination for the predictions JSON.

        Returns
        -------
        Path to the written predictions file.
        """
        from ultralytics import YOLO
        from pycocotools.coco import COCO

        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        model = YOLO(str(ckpt))
        images_dir = Path(images_dir)
        coco_gt = COCO(str(coco_gt_json))
        img_infos = coco_gt.dataset["images"]

        is_mps = self.cfg.device.lower() in _MPS_DEVICES or self.cfg.device.startswith("mps")

        predictions: list[dict] = []
        t0 = time.time()

        for img_info in img_infos:
            img_path = images_dir / img_info["file_name"]
            if not img_path.exists():
                continue

            results = model.predict(
                source=str(img_path),
                imgsz=self.cfg.imgsz,
                conf=0.001,           # very low threshold; COCO AP sweeps over all scores
                device=self.cfg.device,
                verbose=False,
                amp=not is_mps,
            )
            result = results[0]

            if result.masks is None or len(result.boxes) == 0:
                continue

            for box, cls, conf, mask_xy in zip(
                result.boxes.xyxy,
                result.boxes.cls,
                result.boxes.conf,
                result.masks.xy,
            ):
                x1, y1, x2, y2 = box.cpu().numpy().tolist()
                polygon = mask_xy.flatten().tolist()
                if len(polygon) < 6:   # need ≥3 points
                    continue

                predictions.append(
                    {
                        "image_id": img_info["id"],
                        # YOLO cls is 0-based; COCO GT categories are 1-based
                        # (yolo_to_coco: category_id = cls + 1) -> shift by +1,
                        # sonst matcht pycocotools die falsche Kategorie -> mAP~0.
                        "category_id": int(cls.item()) + 1,
                        "bbox": [x1, y1, float(x2 - x1), float(y2 - y1)],
                        "segmentation": [polygon],
                        "score": float(conf.item()),
                    }
                )

        elapsed = time.time() - t0
        n_imgs = len(img_infos)
        _ = elapsed / max(n_imgs, 1) * 1000  # ms per image (unused here, reported by caller)

        out_json = Path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(predictions, indent=2))
        return out_json


# ---------------------------------------------------------------------------
# Module-level predict_to_coco (matches §2.5 interface)
# ---------------------------------------------------------------------------

def predict_to_coco(
    ckpt: Union[str, Path],
    coco_gt_json: Union[str, Path],
    images_dir: Union[str, Path],
    out_json: Union[str, Path],
    cfg: Stage1Config | None = None,
) -> Path:
    """Functional wrapper — create a temporary trainer and run prediction."""
    if cfg is None:
        cfg = Stage1Config()
    trainer = YoloSegTrainer(cfg=cfg, data_yaml="", run_dir=Path(out_json).parent)
    return trainer.predict_to_coco(ckpt, coco_gt_json, images_dir, out_json)
