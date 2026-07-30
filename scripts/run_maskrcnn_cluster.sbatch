#!/bin/bash
# =====================================================================
# SLURM Job-Array: Mask R-CNN Stage-1, 9 Laeufe parallel (je 1 GPU).
#   Index 0-5: real-only (aug on/off x 3 Seeds)   -> Architektur/Augmentierung
#   Index 6-8: synth->real Finetuning (aug-on x 3) -> AP2
# Aufruf (aus dem Repo-Root):   sbatch scripts/run_maskrcnn_cluster.sbatch
# Nebenlaeufigkeit begrenzen (z.B. max 4 gleichzeitig):  --array=0-8%4
# =====================================================================
#SBATCH --job-name=mrcnn100
#SBATCH --array=0-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/mrcnn_%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

# ===================== VOM KOLLEGEN ANPASSEN =========================
# module load cuda/12.6            # falls das Cluster Module nutzt
source /PFAD/ZU/venv-kip/bin/activate     # <-- venv-Pfad eintragen
# =====================================================================

export KIP_WORKERS=0
EPOCHS=100

# --- Split-Guard: nur der korrekte Option-B-Split (deda1092) ---
SHA=$(python -c "import hashlib;print(hashlib.sha256(open('data/coco_converted/test.json','rb').read()).hexdigest()[:12])")
if [ "$SHA" != "deda10926453" ]; then
  echo "ABBRUCH: falscher/fehlender Split (sha=$SHA, erwartet deda10926453, 148 Bilder tool10+tool98)"; exit 1
fi

# --- 9 Konfigurationen; Index = SLURM_ARRAY_TASK_ID ---
AUGS=(on on on off off off on on on)
SEEDS=(42 1 2 42 1 2 42 1 2)
TAGS=(realonly realonly realonly realonly realonly realonly synthpretrain synthpretrain synthpretrain)

i=$SLURM_ARRAY_TASK_ID
AUG=${AUGS[$i]}; SEED=${SEEDS[$i]}; TAG=${TAGS[$i]}

# --- Synth-Laeufe (6-8): Init aus dem Synth-Checkpoint ---
if [ "$TAG" = "synthpretrain" ]; then
  export KIP_MASKRCNN_INIT="$(pwd)/results/maskrcnn_synth/weights/maskrcnn.pt"
  [ -f "$KIP_MASKRCNN_INIT" ] || { echo "ABBRUCH: Synth-Checkpoint fehlt ($KIP_MASKRCNN_INIT)"; exit 1; }
fi

echo "[$(date)] Task $i: model=maskrcnn aug=$AUG seed=$SEED tag=$TAG epochs=$EPOCHS imgsz=800 batch=8"
python scripts/run_stage1.py --model maskrcnn --aug "$AUG" --epochs "$EPOCHS" \
  --imgsz 800 --batch 8 --device cuda:0 --seed "$SEED" --tag "$TAG"
echo "[$(date)] Task $i fertig."
