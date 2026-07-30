"""Mask2Former (Swin-T) fine-tuner and COCO predictor for Stage-1 benchmark.

Design decisions (§2.5 / §6 BUILD_PLAN):
- Plain torch training loop; NO accelerate or HuggingFace Trainer.
- Swin backbone frozen for the first ``cfg.freeze_backbone_epochs`` epochs,
  then unfrozen with a small learning rate (cfg.lr / 10).
- Gradient clipping at 0.1 to stabilise early training.
- Zero-instance images are skipped by the dataset (all 771 train images have
  at least one annotation so this is a no-op in practice).
- Albumentations spatial/colour augmentation applied to image + seg-map
  before calling the HF processor so that the processor's resize/normalise
  pipeline is the authoritative pre-processing for both conditions.
- MPS: PYTORCH_ENABLE_MPS_FALLBACK=1, float32 throughout (no AMP).
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from kip import CLASS_NAMES
from kip.config import Stage1Config


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_ID = "facebook/mask2former-swin-tiny-coco-instance"
_IGNORE_IDX = 255       # background label in segmentation map
_CONF_THRESHOLD = 0.5   # post-processing threshold for instance detection
_GRAD_CLIP = 0.1        # max L2 norm of gradient
# Smoke: max images to load for training / inference
_SMOKE_TRAIN_N = 20
_SMOKE_VAL_N = 10


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CocoInstanceDataset(Dataset):
    """PyTorch Dataset that builds Mask2Former inputs from a COCO JSON.

    Segmentation maps are constructed by rasterising polygon annotations
    (per-instance ID) onto a canvas filled with ``_IGNORE_IDX`` (background).
    The processor converts these maps to per-instance binary mask tensors.

    Parameters
    ----------
    coco_json:
        Path to COCO annotations JSON (train / val / test).
    images_dir:
        Directory that contains the image files referenced by ``file_name``.
    processor:
        ``Mask2FormerImageProcessor`` instance (shared; thread-safe for reads).
    aug_on:
        When True, applies spatial + colour augmentation before the processor.
    max_n:
        If set, only use the first ``max_n`` images (smoke mode).
    """

    def __init__(
        self,
        coco_json: Union[str, Path],
        images_dir: Union[str, Path],
        processor,
        aug_on: bool = False,
        max_n: int | None = None,
    ) -> None:
        with open(coco_json) as f:
            self.coco_data = json.load(f)

        self.images_dir = Path(images_dir)
        self.processor = processor
        self.aug_on = aug_on

        # image_id -> list of annotation dicts
        self._img_to_anns: dict[int, list] = defaultdict(list)
        for ann in self.coco_data["annotations"]:
            self._img_to_anns[ann["image_id"]].append(ann)

        # Only include images that have ≥ 1 annotation
        self.images = [
            img for img in self.coco_data["images"]
            if img["id"] in self._img_to_anns
        ]

        if max_n is not None:
            self.images = self.images[:max_n]

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        img_info = self.images[idx]
        img_path = self.images_dir / img_info["file_name"]
        h, w = img_info["height"], img_info["width"]

        img_np = cv2.imread(str(img_path))
        if img_np is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        anns = self._img_to_anns[img_info["id"]]

        # Build segmentation map: background = _IGNORE_IDX, instances = 1…N
        seg_map = np.full((h, w), _IGNORE_IDX, dtype=np.int32)
        instance_id_to_semantic_id: dict[int, int] = {}

        for inst_id, ann in enumerate(anns, start=1):
            segs = ann.get("segmentation", [])
            if not segs or (isinstance(segs, dict)):
                continue
            for seg in segs:
                pts = np.array(seg, dtype=np.float32).reshape(-1, 2)
                if len(pts) < 3:
                    continue
                cv2.fillPoly(seg_map, [pts.astype(np.int32)], color=inst_id)
            instance_id_to_semantic_id[inst_id] = ann["category_id"]

        # Spatial + colour augmentation (applied before processor)
        if self.aug_on:
            img_np, seg_map = _augment(img_np, seg_map)

        # Processor encodes image + segmentation map -> tensors
        inputs = self.processor(
            images=[Image.fromarray(img_np)],
            segmentation_maps=[seg_map],
            instance_id_to_semantic_id=[instance_id_to_semantic_id],
            ignore_index=_IGNORE_IDX,
            return_tensors="pt",
        )

        return {
            "pixel_values": inputs["pixel_values"][0],       # (3, H, W)
            "pixel_mask": inputs["pixel_mask"][0],           # (H, W)
            "mask_labels": inputs["mask_labels"],            # list[Tensor]
            "class_labels": inputs["class_labels"],          # list[Tensor]
        }


def _augment(
    img_np: np.ndarray,
    seg_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply spatial + colour augmentation to (HWC uint8 image, HW int32 seg_map)."""
    import albumentations as A

    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.GaussNoise(p=0.2),
        ],
        additional_targets={"seg": "mask"},
    )
    result = transform(image=img_np, seg=seg_map)
    return result["image"], result["seg"]


def _collate_fn(batch: list[dict]) -> dict:
    """Custom collate that handles variable-length mask_labels / class_labels."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    pixel_mask = torch.stack([b["pixel_mask"] for b in batch])
    mask_labels = [b["mask_labels"][0] for b in batch]   # list of (N_i, H, W)
    class_labels = [b["class_labels"][0] for b in batch] # list of (N_i,)
    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "mask_labels": mask_labels,
        "class_labels": class_labels,
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Mask2FormerTrainer:
    """Fine-tune ``facebook/mask2former-swin-tiny-coco-instance`` on custom data.

    Parameters
    ----------
    cfg:
        Stage1Config controlling epochs, batch, lr, freeze_backbone_epochs,
        device, seed, smoke, augmentation.
    coco_train_json:
        COCO annotations JSON for the training split.
    coco_val_json:
        COCO annotations JSON for the validation split (not used for early
        stopping here; reserved for future use).
    images_dir:
        Root directory containing all images (train + val images must both be
        findable under this single directory).
    run_dir:
        Output directory.  Checkpoints land in ``<run_dir>/checkpoints/``.
    """

    def __init__(
        self,
        cfg: Stage1Config,
        coco_train_json: Union[str, Path],
        coco_val_json: Union[str, Path],
        images_dir: Union[str, Path],
        run_dir: Path,
    ) -> None:
        self.cfg = cfg
        self.coco_train_json = Path(coco_train_json)
        self.coco_val_json = Path(coco_val_json)
        self.images_dir = Path(images_dir)
        self.run_dir = Path(run_dir)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> Path:
        """Fine-tune Mask2Former and return the checkpoint directory path.

        The checkpoint directory contains ``pytorch_model.bin`` (state dict)
        and ``config.json`` so that ``from_pretrained`` can reload it.
        """
        from transformers import (
            Mask2FormerForUniversalSegmentation,
            Mask2FormerImageProcessor,
        )

        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        # --- resolve device ---
        device = torch.device(self.cfg.device)

        # --- processor (shared between train and predict) ---
        processor = Mask2FormerImageProcessor.from_pretrained(_MODEL_ID)

        # --- dataset ---
        max_n = _SMOKE_TRAIN_N if self.cfg.smoke else None
        train_ds = CocoInstanceDataset(
            coco_json=self.coco_train_json,
            images_dir=self.images_dir,
            processor=processor,
            aug_on=self.cfg.augmentation,
            max_n=max_n,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch,
            shuffle=True,
            collate_fn=_collate_fn,
            num_workers=0,      # keep simple; avoids multiprocess tensor issues
            pin_memory=False,
        )

        # --- model ---
        # id2label / label2id covering our 9 classes (IDs 0-8)
        id2label = {i: name for i, name in enumerate(CLASS_NAMES)}
        label2id = {v: k for k, v in id2label.items()}

        _init_id = os.environ.get("KIP_M2F_INIT", _MODEL_ID)
        print(f"[M2F] Initialisierung von: {_init_id}")
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            _init_id,
            num_labels=len(CLASS_NAMES),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        model.to(device)

        # --- freeze Swin backbone initially ---
        if self.cfg.freeze_backbone_epochs > 0:
            _set_backbone_grad(model, requires_grad=False)

        # --- optimiser: two param groups for backbone / head ---
        backbone_params = list(model.model.pixel_level_module.encoder.parameters())
        head_params = [
            p for p in model.parameters()
            if not any(p is bp for bp in backbone_params)
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": head_params, "lr": self.cfg.lr},
                {"params": backbone_params, "lr": self.cfg.lr / 10},
            ],
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, self.cfg.epochs)
        )

        # --- training loop ---
        best_loss = float("inf")
        ckpt_dir = self.run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        t_train_start = time.time()

        for epoch in range(self.cfg.epochs):
            # Unfreeze backbone after freeze_backbone_epochs
            if (
                self.cfg.freeze_backbone_epochs > 0
                and epoch == self.cfg.freeze_backbone_epochs
            ):
                _set_backbone_grad(model, requires_grad=True)

            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                batch = _to_device(batch, device)
                optimizer.zero_grad()

                outputs = model(
                    pixel_values=batch["pixel_values"],
                    pixel_mask=batch["pixel_mask"],
                    mask_labels=batch["mask_labels"],
                    class_labels=batch["class_labels"],
                )
                loss = outputs.loss
                if loss is None or not torch.isfinite(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), _GRAD_CLIP)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"[M2F epoch {epoch+1}/{self.cfg.epochs}] loss={avg_loss:.4f}")

            # Save best checkpoint by training loss
            if avg_loss < best_loss:
                best_loss = avg_loss
                model.save_pretrained(ckpt_dir)
                processor.save_pretrained(ckpt_dir)

        # Guarantee at least one checkpoint
        if not (ckpt_dir / "config.json").exists():
            model.save_pretrained(ckpt_dir)
            processor.save_pretrained(ckpt_dir)

        train_seconds = time.time() - t_train_start
        print(
            f"[M2F] Training done in {train_seconds:.1f}s. "
            f"Best loss: {best_loss:.4f}. Checkpoint: {ckpt_dir}"
        )
        return ckpt_dir

    # ------------------------------------------------------------------
    # Prediction -> COCO
    # ------------------------------------------------------------------

    def predict_to_coco(
        self,
        ckpt_dir: Union[str, Path],
        coco_gt_json: Union[str, Path],
        images_dir: Union[str, Path],
        out_json: Union[str, Path],
    ) -> Path:
        """Run Mask2Former inference and write COCO-format predictions.

        Parameters
        ----------
        ckpt_dir:
            Directory produced by ``train()`` (contains config.json,
            pytorch_model.bin / model.safetensors).
        coco_gt_json:
            COCO GT JSON (image list source).
        images_dir:
            Directory containing image files.
        out_json:
            Destination for COCO predictions JSON.

        Returns
        -------
        Path to the written predictions file.
        """
        from transformers import (
            Mask2FormerForUniversalSegmentation,
            Mask2FormerImageProcessor,
        )
        from pycocotools.coco import COCO

        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        device = torch.device(self.cfg.device)

        processor = Mask2FormerImageProcessor.from_pretrained(str(ckpt_dir))
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            str(ckpt_dir),
            ignore_mismatched_sizes=False,
        )
        model.to(device)
        model.eval()

        coco_gt = COCO(str(coco_gt_json))
        img_infos = coco_gt.dataset["images"]
        images_dir = Path(images_dir)

        predictions: list[dict] = []

        with torch.no_grad():
            for img_info in img_infos:
                img_path = images_dir / img_info["file_name"]
                if not img_path.exists():
                    continue

                orig_h = img_info["height"]
                orig_w = img_info["width"]

                img = Image.open(img_path).convert("RGB")
                inputs = processor(images=[img], return_tensors="pt")
                inputs = _to_device(inputs, device)

                outputs = model(pixel_values=inputs["pixel_values"])

                # Post-process to original image size
                results = processor.post_process_instance_segmentation(
                    outputs,
                    threshold=_CONF_THRESHOLD,
                    target_sizes=[(orig_h, orig_w)],
                    return_binary_maps=True,
                )
                res = results[0]

                seg_maps = res.get("segmentation")  # Tensor (N, H, W) or None
                segs_info = res.get("segments_info", [])

                if seg_maps is None or len(segs_info) == 0:
                    continue

                # seg_maps may be a list when return_binary_maps=True
                if isinstance(seg_maps, torch.Tensor):
                    seg_maps = seg_maps.cpu().numpy()   # (N, H, W) float/bool
                elif isinstance(seg_maps, list):
                    seg_maps = [
                        s.cpu().numpy() if isinstance(s, torch.Tensor) else np.array(s)
                        for s in seg_maps
                    ]
                    if not seg_maps:
                        continue
                    seg_maps = np.stack(seg_maps, axis=0)

                for i, info in enumerate(segs_info):
                    if i >= len(seg_maps):
                        break
                    binary_mask = (seg_maps[i] > 0.5).astype(np.uint8)
                    if binary_mask.sum() == 0:
                        continue

                    # Convert binary mask to polygon
                    polygon = _mask_to_polygon(binary_mask)
                    if polygon is None:
                        continue

                    # Bounding box from mask
                    bbox = _mask_to_bbox(binary_mask)

                    predictions.append(
                        {
                            "image_id": img_info["id"],
                            "category_id": int(info["label_id"]),
                            "bbox": bbox,
                            "segmentation": [polygon],
                            "score": float(info["score"]),
                        }
                    )

        out_json = Path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(predictions, indent=2))
        return out_json


# ---------------------------------------------------------------------------
# Module-level predict_to_coco (matches §2.5 interface)
# ---------------------------------------------------------------------------

def predict_to_coco(
    ckpt_dir: Union[str, Path],
    coco_gt_json: Union[str, Path],
    images_dir: Union[str, Path],
    out_json: Union[str, Path],
    cfg: Stage1Config | None = None,
) -> Path:
    """Functional wrapper — create a temporary trainer and run prediction."""
    if cfg is None:
        cfg = Stage1Config()
    trainer = Mask2FormerTrainer(
        cfg=cfg,
        coco_train_json=coco_gt_json,
        coco_val_json=coco_gt_json,
        images_dir=images_dir,
        run_dir=Path(out_json).parent,
    )
    return trainer.predict_to_coco(ckpt_dir, coco_gt_json, images_dir, out_json)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_backbone_grad(model, requires_grad: bool) -> None:
    """Freeze or unfreeze the Swin encoder backbone."""
    for param in model.model.pixel_level_module.encoder.parameters():
        param.requires_grad_(requires_grad)


def _to_device(obj, device: torch.device):
    """Recursively move tensors (or dict/list of tensors) to device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, Mapping):  # dict AND transformers BatchFeature (UserDict)
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_device(v, device) for v in obj]
    return obj


def _mask_to_polygon(binary_mask: np.ndarray) -> list[float] | None:
    """Extract the largest contour from a binary mask as a flat polygon list."""
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None
    return contour.flatten().tolist()


def _mask_to_bbox(binary_mask: np.ndarray) -> list[float]:
    """Return [x, y, w, h] bounding box from a binary mask."""
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [float(cmin), float(rmin), float(cmax - cmin), float(rmax - rmin)]
