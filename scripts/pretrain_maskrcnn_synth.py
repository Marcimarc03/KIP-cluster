#!/usr/bin/env python
"""Mask R-CNN auf dem SYNTHETISCHEN Datensatz vortrainieren (Strategie-C-Basis).

Analog zu ``pretrain_m2f_synth.py``: erzeugt einen leakage-freien Checkpoint
(nur synthetische Daten, kein reales Werkzeug), der anschliessend ueber die
Umgebungsvariable ``KIP_MASKRCNN_INIT`` als Initialisierung fuer das reale
Finetuning dient. Damit ist die Synthetik-Ablation fuer Mask R-CNN symmetrisch
zu YOLO (--weights A_synth_only) und M2F (KIP_M2F_INIT).

Konfiguration analog zu den realen Mask-R-CNN-Laeufen (imgsz 800, batch/epochs
per CLI). Einziger Unterschied: Trainingsdaten = data/coco_synth statt real.

Ausgabe:  results/maskrcnn_synth/weights/maskrcnn.pt

Nutzung (in tmux, venv-kip aktiv, auf der DGX wo data/coco_synth liegt):
    python scripts/pretrain_maskrcnn_synth.py --epochs 100 --batch 4 --device cuda:0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kip.config import Stage1Config, seed_everything
from kip.stage1.maskrcnn_trainer import MaskRCNNTrainer

_SYNTH_COCO = _ROOT / "data" / "coco_synth"
_SYNTH_IMAGES = _ROOT / "data" / "synth_Daten" / "images" / "train"
_OUT = _ROOT / "results" / "maskrcnn_synth"


def main() -> None:
    ap = argparse.ArgumentParser(description="Mask R-CNN synth-Vortraining")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--aug", choices=["on", "off"], default="on")
    args = ap.parse_args()

    train_json = _SYNTH_COCO / "train.json"
    val_json = _SYNTH_COCO / "val.json"
    for p in (train_json, val_json, _SYNTH_IMAGES):
        if not p.exists():
            sys.exit(f"ERROR: fehlt: {p}  (Synth-COCO nur auf der DGX vorhanden)")

    seed_everything(args.seed)
    _OUT.mkdir(parents=True, exist_ok=True)

    cfg = Stage1Config(
        model="maskrcnn",
        augmentation=(args.aug == "on"),
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        smoke=False,
    )

    print(f"[pretrain] synth-COCO : {train_json}")
    print(f"[pretrain] Bilder     : {_SYNTH_IMAGES}")
    print(f"[pretrain] Ausgabe    : {_OUT}")
    print(f"[pretrain] epochs={args.epochs} batch={args.batch} imgsz={args.imgsz} "
          f"seed={args.seed} aug={args.aug}")

    trainer = MaskRCNNTrainer(
        cfg=cfg,
        coco_train_json=train_json,
        coco_val_json=val_json,
        images_dir=_SYNTH_IMAGES,
        run_dir=_OUT,
        label_offset=-1,   # Synth ist 1-indiziert -> auf 0-indizierten Real-Klassenraum bringen
    )

    t0 = time.time()
    ckpt = trainer.train()
    print(f"[pretrain] fertig in {(time.time()-t0)/3600:.2f} h")
    print(f"[pretrain] CHECKPOINT: {ckpt}")
    print("[pretrain] Fuer das Finetuning:")
    print(f"    export KIP_MASKRCNN_INIT={ckpt}")


if __name__ == "__main__":
    main()
