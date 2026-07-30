# KIP — Mask R-CNN Stage-1 (Cluster-Paket)

Minimales Paket, um die Mask-R-CNN-Stage-1-Laeufe (100 Epochen, val-basierte
Checkpoint-Auswahl gegen Overfitting) auf dem AIFB-Cluster zu fahren.

## 1. Umgebung
```bash
python -m venv venv-kip && source venv-kip/bin/activate
# torch/torchvision passend zur Cluster-CUDA installieren, dann:
pip install -r requirements-cluster.txt
```

## 2. Daten bereitstellen (NICHT im git — separat kopieren)
Genau diese Pfade werden gelesen (relativ zum Repo-Root):
```
data/coco_converted/train.json          # Option-B-Split (sha deda1092)
data/coco_converted/val.json
data/coco_converted/test.json
data/object_segmentation_real_v3_1088/images/train/    # 713 Bilder
data/object_segmentation_real_v3_1088/images/val/      # 101 Bilder (fuer best-Checkpoint)
data/object_segmentation_real_v3_1088/images/test/     # 148 Bilder
data/object_segmentation_real_v3_1088/data.yaml        # wird beim Start geprueft
```
Fuer die Synth-Laeufe (Index 6-8) zusaetzlich:
```
results/maskrcnn_synth/weights/maskrcnn.pt             # Synth-Checkpoint (~350 MB, von der DGX)
```
**Split-Pruefstein:** `test.json` muss den sha `deda10926453` haben (148 Bilder,
Werkzeuge tool10+tool98). Das sbatch-Skript prueft das und bricht sonst ab.

## 3. Starten (SLURM Job-Array, 9 Laeufe parallel)
In `scripts/run_maskrcnn_cluster.sbatch` oben den venv-Pfad (und ggf. `module load`)
eintragen, dann:
```bash
sbatch scripts/run_maskrcnn_cluster.sbatch          # alle 9 gleichzeitig
# oder Nebenlaeufigkeit begrenzen:
sbatch --array=0-8%4 scripts/run_maskrcnn_cluster.sbatch
```
Nur Pflicht (real, 100-Epochen-Paritaet): `--array=0-5`. Synth (Kuer): `--array=6-8`.

## 4. Auswertung
```bash
python scripts/aggregate_stage1.py     # Mittel +/- Std je Modell/Config
```
Die metrics.json der neuen Laeufe liegen unter `results/component_benchmark/`.

## Hinweise
- **Val-basierte Auswahl:** Der Trainer bewertet alle 5 Epochen das Val-Set und
  speichert das beste Modell -> 100 Epochen sind overfitting-sicher, die maximale
  Epochenzahl ist unkritisch. Frequenz via `KIP_MRCNN_VAL_EVERY` (Default 5).
- **Hardware-Note fuers Paper:** Diese Laeufe liefen auf dem Cluster (andere GPU
  als die DGX-YOLO/M2F-Laeufe); als Multi-Seed-Mittel berichtet, absorbiert die
  Streuung.
