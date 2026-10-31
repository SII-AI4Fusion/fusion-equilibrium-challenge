#!/usr/bin/env bash
# FAIR-MAST download -> training cache -> sensor health audit (all CPU).
#
#   bash scripts/prepare_mast_data.sh                       full archive (11 573 shots)
#   SHOTS="26497 27000" bash scripts/prepare_mast_data.sh   a few shots (smoke test)
#
# Environment: RAW_DIR [data/mast_data_v3] (FAIR-MAST Zarr stores), CACHE_DIR [data/crossmak-mast-cache-v3],
#              HEALTH_DIR [data/health_v1], WORKERS [16], SKIP_DOWNLOAD=1 to use an existing RAW_DIR.
set -euo pipefail
source "$(dirname "$0")/env.sh"
RAW_DIR=${RAW_DIR:-$REPO/data/mast_data_v3}
CACHE_DIR=${CACHE_DIR:-$REPO/data/crossmak-mast-cache-v3}
HEALTH_DIR=${HEALTH_DIR:-$REPO/data/health_v1}
WORKERS=${WORKERS:-16}
cd "$REPO/mast"
export PYTHONPATH="$REPO/mast/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ -z "${SKIP_DOWNLOAD:-}" ]; then
    if [ -n "${SHOTS:-}" ]; then
        "$DATA_PY" "$REPO/data/download_fair_mast.py" --dest "$RAW_DIR" --shots $SHOTS
    else
        "$DATA_PY" "$REPO/data/download_fair_mast.py" --dest "$RAW_DIR" --shots-file configs/data/fair_mast_shots.txt
    fi
fi
mkdir -p "$CACHE_DIR"
# shot-level split: hash of the shot number (seed "crossmak-mast-v1", 5 % val, 5 % test)
[ -f "$CACHE_DIR/manifest.json" ] || "$DATA_PY" -m data_prep.scan --config configs/data/mast_fair_v3.json \
    --data-root "$RAW_DIR" --output "$CACHE_DIR/manifest.json" --threads 8
"$DATA_PY" -m data_prep.build_cache --config configs/data/mast_fair_v3.json --cache "$CACHE_DIR" --workers "$WORKERS"
"$DATA_PY" -m data_prep.stats --cache "$CACHE_DIR" --workers "$WORKERS"
"$TORCH_PY" -m data_prep.health features --cache "$CACHE_DIR" --output "$HEALTH_DIR" --workers "$WORKERS"
"$TORCH_PY" -m data_prep.health run --cache "$CACHE_DIR" --output "$HEALTH_DIR" --workers "$WORKERS"
echo "cache: $CACHE_DIR  health: $HEALTH_DIR"
