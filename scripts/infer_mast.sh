#!/usr/bin/env bash
# MAST inference: Final parquet inputs -> mast_private_test.npz.
#
#   v3 network (bf16 on GPU) predicts the 92 magnetics_slow channels and 90 pf_passive currents
#   -> frozen-v4 GSFit, dof=2 + diag regularisation x150 + upstream initial boundary + OBR02 weight x2 + R grid 0.06-2.0;
#      frames that do not converge fall back to the dof=1 solve of the same inputs
#   -> passive-current GSFit on every 8th frame, applied as a correction (PASSIVE_VARIANT, below)
#   -> betaN replaced by the dof=1 betaN (scalar swap only).
#
#   bash scripts/infer_mast.sh all      single node: every step below in order
#   bash scripts/infer_mast.sh prepare  parquet -> per-shot inputs + case manifest (CPU, minutes)
#   bash scripts/infer_mast.sh infer    v3 network on NGPU GPUs, one shard per GPU, then merge
#   bash scripts/infer_mast.sh stage    dof1 / dof2 GSFit run directories (seconds)
#   bash scripts/infer_mast.sh solve    GSFit solves for shard RANK of SHARDS (one CPU node per rank);
#                                       rank 0 then waits for the other ranks and runs finish
#   bash scripts/infer_mast.sh finish   pack, fallback, passive correction, betaN swap
#
# Environment (defaults in brackets):
#   FINAL_DIR   challenge Final dataset root, containing data/mast_private_test/*.parquet  [data/challenge-final]
#   WEIGHTS     directory with best.pt, layout.json, stats.json                         [weights/mast_v3]
#   OUT         run directory                                                            [runs/mast_infer]
#   NGPU [all visible]  DEVICE [cuda]  SHARDS [1]  RANK [0]
#   WORKERS_DOF1 / WORKERS_DOF2 / WORKERS_PASSIVE  GSFit processes per node [40 / 60 / 28, for 128 cores]
#   LIMIT_SHOTS  only the first N shots (smoke test) [0 = all 1205]
#   PASSIVE_VARIANT  delta_all [default]: full-grid correction, no competition asset;
#                    delta_mask: correction restricted to the starter kit's scoring-envelope mask, which
#                    reproduces submission 1012261 exactly; this use of that file outside the official
#                    scoring may be problematic under rule 3, so it is kept for reproduction (see README)
#   TORCH_PY / GSFIT_PY  interpreters of envs/torch and envs/gsfit (see README)
set -uo pipefail
source "$(dirname "$0")/env.sh"
STEP=${1:?usage: infer_mast.sh all|prepare|infer|stage|solve|finish}
FINAL_DIR=${FINAL_DIR:-$REPO/data/challenge-final}
WEIGHTS=${WEIGHTS:-$REPO/weights/mast_v3}
OUT=${OUT:-$REPO/runs/mast_infer}
DEVICE=${DEVICE:-cuda}
N=${SHARDS:-1}
RANK=${RANK:-0}
LIMIT=${LIMIT_SHOTS:-0}
VARIANT=${PASSIVE_VARIANT:-delta_all}
case $VARIANT in delta_all) ENV_ARG=() ;; delta_mask) ENV_ARG=(--envelope "$REPO/mast/assets/mast_envelope.npz") ;;
    *) echo "PASSIVE_VARIANT must be delta_all or delta_mask" >&2; exit 2 ;; esac
RMAX=2.0
PROF=(--dof 2 --reg-form diag --reg-scale 150 --init upstream)
SRC=$OUT/v3 DOF1=$OUT/dof1 DOF2=$OUT/dof2 P=$OUT/passive LOG=$OUT/logs
CONFIG=$REPO/mast/configs/challenge/passive_final.json
V4W=("$GSFIT_PY" "$REPO/mast/scripts/challenge_v4_weighted.py")
PAS=("$GSFIT_PY" -m training.challenge_passive)
mkdir -p "$LOG"
cd "$REPO/mast"
export PYTHONPATH="$REPO/mast/src${GSFIT_RS_DIR:+:$GSFIT_RS_DIR}"
echo "$STEP rank $RANK/$N start $(date -Is) on $(hostname)"

prepare() {
    local extra=()
    [ "$LIMIT" -gt 0 ] && extra=(--n-shots "$LIMIT")
    "$TORCH_PY" -m training.challenge_mast --output "$OUT/input" --parquet "$FINAL_DIR/data/mast_private_test" \
        prepare "${extra[@]}" > "$LOG/prepare.log" 2>&1 || { echo "FAILED prepare"; exit 1; }
    cp "$REPO/mast/assets/mast_static_geometry.npz" "$OUT/input/"
}

infer() {
    local n=${NGPU:-$("$TORCH_PY" -c "import torch; print(max(torch.cuda.device_count(), 1))")} fail=0 pids=()
    for r in $(seq 0 $((n-1))); do
        CUDA_VISIBLE_DEVICES=$r OMP_NUM_THREADS=4 "$TORCH_PY" -W ignore -m training.challenge_v3 infer --source "$OUT/input" \
            --checkpoint "$WEIGHTS/best.pt" --cache-dir "$WEIGHTS" --output "$SRC" --core top5 --p3 zero \
            --device "$DEVICE" --batch-size 64 --rank $r --shards $n > "$LOG/infer_r$r.log" 2>&1 & pids+=($!)
    done
    for p in "${pids[@]}"; do wait $p || fail=1; done
    [ $fail = 0 ] || { echo "FAILED infer"; exit 1; }
    if [ "$n" -gt 1 ]; then
        "$TORCH_PY" -m training.challenge_v3 merge --source x --checkpoint "$WEIGHTS/best.pt" --output "$SRC" --shards $n \
            > "$LOG/infer_merge.log" 2>&1 || { echo "FAILED merge"; exit 1; }
    fi
}

stage() {
    "${V4W[@]}" stage --input "$SRC" --output "$DOF1" --channel-weight OBR02=2 --r-max $RMAX \
        > "$LOG/stage_dof1.log" 2>&1 || { echo "FAILED stage dof1"; exit 1; }
    "${V4W[@]}" stage --input "$SRC" --output "$DOF2" --channel-weight OBR02=2 --r-max $RMAX "${PROF[@]}" \
        > "$LOG/stage_dof2.log" 2>&1 || { echo "FAILED stage dof2"; exit 1; }
}

solve() {
    local shots ok=1 pids=()
    shots=$("$GSFIT_PY" - "$SRC/manifest.json" "$RANK" "$N" <<'EOF'
import json, sys
shots = sorted({int(c["official_index"]) for c in json.load(open(sys.argv[1]))["cases"]})
print(",".join(str(s) for s in shots[int(sys.argv[2])::int(sys.argv[3])]))
EOF
)
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
    "${V4W[@]}" solve --output "$DOF1" --rank "$RANK" --shards $N --workers ${WORKERS_DOF1:-40} --phase job \
        > "$LOG/dof1_r$RANK.log" 2>&1 & pids+=($!)
    "${V4W[@]}" solve --output "$DOF2" --rank "$RANK" --shards $N --workers ${WORKERS_DOF2:-60} --phase job \
        > "$LOG/dof2_r$RANK.log" 2>&1 & pids+=($!)
    "${PAS[@]}" solve --config "$CONFIG" --source "$SRC" --output "${P}_r$RANK" --shots "$shots" \
        --workers ${WORKERS_PASSIVE:-28} --channel-weight OBR02=2 --r-max $RMAX "${PROF[@]}" --passive-source v3 \
        > "$LOG/passive_r$RANK.log" 2>&1 & pids+=($!)
    for p in "${pids[@]}"; do wait $p || ok=0; done
    [ $ok = 1 ] || { touch "$LOG/rank$RANK.failed"; echo "FAILED solve r$RANK"; exit 1; }
    touch "$LOG/rank$RANK.done"; echo "rank $RANK solve done $(date -Is)"
    [ "$RANK" = 0 ] || exit 0
    for r in $(seq 0 $((N-1))); do
        until [ -f "$LOG/rank$r.done" ]; do
            [ -f "$LOG/rank$r.failed" ] && { echo "rank $r failed"; exit 1; }
            sleep 15
        done
    done
    finish
}

merge_passive() {
    mkdir -p "$P/shots_gsfit"
    for r in $(seq 0 $((N-1))); do ln -sf "${P}_r$r/shots_gsfit/"*.npz "$P/shots_gsfit/"; done
    "$GSFIT_PY" - "$P" "$N" <<'EOF'
import json, sys
out, n = sys.argv[1], int(sys.argv[2])
parts = [json.load(open(f"{out}_r{r}/gsfit_summary.json")) for r in range(n)]
merged = dict(parts[0])
for k in ("shots", "total_frames", "attempted_frames", "converged_frames", "neg_p_frames", "ts_recovered_frames"):
    merged[k] = sum(p[k] for p in parts)
merged["failed_shots"] = sorted(s for p in parts for s in p["failed_shots"])
merged["elapsed_seconds"] = max(p["elapsed_seconds"] for p in parts)
merged["per_shot"] = sorted((row for p in parts for row in p["per_shot"]), key=lambda row: row["shot"])
json.dump(merged, open(f"{out}/gsfit_summary.json", "w"), indent=1)
print("merged", merged["shots"], "shots, converged", merged["converged_frames"], "/", merged["attempted_frames"])
EOF
}

finish() {
    "${V4W[@]}" pack --output "$DOF1" > "$LOG/dof1_pack.log" 2>&1 || { echo "FAILED dof1 pack"; exit 1; }
    "${V4W[@]}" pack --output "$DOF2" --fallback "$DOF1" > "$LOG/dof2_pack.log" 2>&1 || { echo "FAILED dof2 pack"; exit 1; }
    merge_passive
    "${PAS[@]}" pack --config "$CONFIG" --source "$SRC" --output "$P" --baseline-submission "$DOF2/submission_mast.npz" \
        --channel-weight OBR02=2 "${ENV_ARG[@]}" > "$LOG/passive_pack.log" 2>&1 || { echo "FAILED passive pack"; exit 1; }
    # betaN from the dof=1 solve (same inputs, same grid); only the scalar is swapped.
    "$GSFIT_PY" - "$P/submission_$VARIANT.npz" "$DOF1/submission_mast.npz" "$OUT/mast_private_test.npz" <<'EOF'
import sys
import numpy as np
src, dof1, dst = sys.argv[1:]
with np.load(src) as a, np.load(dof1) as b:
    for k in a.files:
        if k.endswith("_betaN"):
            assert b[k].shape == a[k].shape and np.isfinite(b[k]).all(), k
    out = {k: (b[k] if k.endswith("_betaN") else a[k]) for k in a.files}
np.savez_compressed(dst, **out)
print("betaN from dof1:", sum(k.endswith("_betaN") for k in out), "shots")
EOF
    [ $? = 0 ] || { echo "FAILED betaN swap"; exit 1; }
    echo "wrote $OUT/mast_private_test.npz ($VARIANT)"
}

case $STEP in
    prepare) prepare ;;
    infer) infer ;;
    stage) stage ;;
    solve) solve ;;
    finish) finish ;;
    all) prepare; infer; stage; N=1; RANK=0; solve ;;
    *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
echo "$STEP done $(date -Is)"
