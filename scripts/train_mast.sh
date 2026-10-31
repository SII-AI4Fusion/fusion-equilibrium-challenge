#!/usr/bin/env bash
# Train the MAST v3 network from scratch (supervised: Ip, PF/SOL, TF, Thomson -> magnetics_slow + pf_passive).
#
#   bash scripts/train_mast.sh            one GPU (the declared setting: single 80 GB A100-class GPU)
#   NPROC=8 bash scripts/train_mast.sh    8 GPUs, as the submitted weights were trained (8 x H100, 1.7 h)
#
# The global batch is always 256 windows (256 / NPROC per GPU), 40 000 steps, bf16; the checkpoint with
# the lowest validation loss (validation = the hash-split val shots of the cache) is kept as best.pt.
# Environment: CACHE_DIR [data/crossmak-mast-cache-v3], HEALTH_DIR [data/health_v1], OUT [runs/mast_train]
set -euo pipefail
source "$(dirname "$0")/env.sh"
NPROC=${NPROC:-1}
CACHE_DIR=$(realpath "${CACHE_DIR:-$REPO/data/crossmak-mast-cache-v3}")
HEALTH_DIR=$(realpath "${HEALTH_DIR:-$REPO/data/health_v1}")
OUT=$(realpath -m "${OUT:-$REPO/runs/mast_train}")
(( 256 % NPROC == 0 )) || { echo "NPROC must divide 256" >&2; exit 2; }
cd "$REPO/mast"
export PYTHONPATH="$REPO/mast/src"
"$TORCH_PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" -m training.train \
    --config configs/train/v3.json \
    --set data.cache_dir="$CACHE_DIR" --set data.health_dir="$HEALTH_DIR" --set run.output_dir="$OUT" \
    --set optim.batch_size=$((256 / NPROC))
# inference needs the cache normalisation next to the weights
cp "$CACHE_DIR/layout.json" "$CACHE_DIR/stats.json" "$OUT/checkpoints/"
echo "weights: $OUT/checkpoints/best.pt (+ layout.json, stats.json); use WEIGHTS=$OUT/checkpoints for infer_mast.sh"
