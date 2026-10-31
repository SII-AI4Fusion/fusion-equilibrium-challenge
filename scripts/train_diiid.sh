#!/usr/bin/env bash
# DIII-D: feature cache from the official diii_d_train split, then SimpleMLP (psi) + RidgeCV (q95, betaN).
# One GPU, about 6 minutes of training after a CPU cache build.
# Environment: TRAIN_DIR [data/challenge-v1.1.0] (contains data/diii_d_train), CACHE_DIR [data/diiid_cache],
#              OUT [runs/diiid_train], WORKERS [8]
set -euo pipefail
source "$(dirname "$0")/env.sh"
TRAIN_DIR=${TRAIN_DIR:-$REPO/data/challenge-v1.1.0}
CACHE_DIR=${CACHE_DIR:-$REPO/data/diiid_cache}
OUT=${OUT:-$REPO/runs/diiid_train}
export OMP_NUM_THREADS=8
[ -f "$CACHE_DIR/COMPLETE.json" ] || "$TORCH_PY" "$REPO/diiid/scripts/build_diiid_cache.py" \
    --data "$TRAIN_DIR" --out "$CACHE_DIR" --workers "${WORKERS:-8}"
"$TORCH_PY" "$REPO/diiid/scripts/train_diiid_simple_mlp.py" train --pca-run "$CACHE_DIR" --out "$OUT" --epochs 50
echo "weights: $OUT/model.pt"
