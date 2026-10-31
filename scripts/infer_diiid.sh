#!/usr/bin/env bash
# DIII-D inference: Final diii_d_private_test parquet -> diii_d_private_test.npz (psirz float16, q95/betaN float32).
# Environment: FINAL_DIR [data/challenge-final], WEIGHTS [weights/diiid] (model.pt), OUT [runs/diiid_infer], DEVICE [cuda]
set -euo pipefail
source "$(dirname "$0")/env.sh"
FINAL_DIR=${FINAL_DIR:-$REPO/data/challenge-final}
WEIGHTS=${WEIGHTS:-$REPO/weights/diiid}
OUT=${OUT:-$REPO/runs/diiid_infer}
"$TORCH_PY" "$REPO/diiid/scripts/train_diiid_simple_mlp.py" predict --out "$WEIGHTS" \
    --input-dir "$FINAL_DIR/data/diii_d_private_test" --output "$OUT/diii_d_private_test.npz" \
    --device "${DEVICE:-cuda}" ${LIMIT_SHOTS:+--max-shots $LIMIT_SHOTS}
