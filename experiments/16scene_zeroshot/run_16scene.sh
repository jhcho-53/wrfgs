#!/usr/bin/env bash
# Reproduce the 16-scene scale-up experiment: train the relative-geometry
# scene-conditioned WRF-GS+ model on 12 V2I scenarios and evaluate zero-shot on
# 4 held-out scenes (one per town). See docs/research/2026-06-18-...md §4.4 and
# experiments/16scene_zeroshot/README.md.
#
# Prereqs (must already be present on disk; both are .gitignored):
#   - data_mw_<Town>_<scenario>/  : the 16 converted datasets (tools/mw2wrfgs)
#   - scenes_static/<scene>.npy   : the 16 static lidar clouds (tools/extract_rsu_scene.py)
#
# Usage:  bash experiments/16scene_zeroshot/run_16scene.sh [GPU] [ITERS]
set -euo pipefail

GPU="${1:-0}"
ITERS="${2:-12000}"
SEED="${SEED:-0}"
PY="${PYTHON:-python}"   # set PYTHON=/path/to/wrfgsplus/bin/python if `conda activate` is unavailable

# Run from the repo root regardless of where this is invoked.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="experiments/16scene_zeroshot/runs/retrain_seed${SEED}_${STAMP}"
mkdir -p "$OUT"
LOG="$OUT/train.log"

# 12 training scenes (4 towns; INCLUDES elevated-RSU scenes Town03_Tjunction z=8m,
# Town03_crossroad z=8.2m so the model sees elevated geometry).
TRAIN=(
  data_mw_Town03_crossroad:scenes_static/Town03_crossroad.npy
  data_mw_Town03_gastation:scenes_static/Town03_gastation.npy
  data_mw_Town03_roundabout:scenes_static/Town03_roundabout.npy
  data_mw_Town03_Tjunction:scenes_static/Town03_Tjunction.npy
  data_mw_Town05_2skybridge:scenes_static/Town05_2skybridge.npy
  data_mw_Town05_CBDcrossroad:scenes_static/Town05_CBDcrossroad.npy
  data_mw_town05_parkinglot:scenes_static/Town05_parkinglot.npy
  data_mw_Town05_Tjunction:scenes_static/Town05_Tjunction.npy
  data_mw_Town07_crossroad:scenes_static/Town07_crossroad.npy
  data_mw_Town10_curvyroad:scenes_static/Town10_curvyroad.npy
  data_mw_Town10_Hroad:scenes_static/Town10_Hroad.npy
  data_mw_Town10_skybridge:scenes_static/Town10_skybridge.npy
)
# 4 held-out scenes, one per town, NEVER trained (zero-shot).
HOLDOUT=(
  data_mw_Town03_5wayroad:scenes_static/Town03_5wayroad.npy
  data_mw_Town05_ringroad:scenes_static/Town05_ringroad.npy
  data_mw_Town07_grainsilos:scenes_static/Town07_grainsilos.npy
  data_mw_Town10_crossroad:scenes_static/Town10_crossroad.npy
)

echo "[run_16scene] GPU=$GPU iters=$ITERS seed=$SEED out=$OUT py=$PY"
# train_multiscene.py has no --gpu flag; it uses .cuda() (device 0 of visible set).
CUDA_VISIBLE_DEVICES="$GPU" "$PY" train_multiscene.py \
  --iterations "$ITERS" --seed "$SEED" \
  --out-dir "$OUT" --eval-every 4000 \
  --scenes "${TRAIN[@]}" \
  --holdout-scenes "${HOLDOUT[@]}" 2>&1 | tee "$LOG"

echo "[run_16scene] checkpoint + results -> $OUT"
echo "[run_16scene] reload & re-evaluate the trained model with:"
echo "  python train_multiscene.py --eval-only --load-checkpoint $OUT/deform_latest.pth \\"
echo "    --out-dir $OUT/eval --scenes ${TRAIN[*]} --holdout-scenes ${HOLDOUT[*]}"
