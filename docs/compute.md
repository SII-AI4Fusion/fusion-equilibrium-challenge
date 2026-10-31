# Training compute declaration

## GPU training (both machines together)

| Step | Hardware used for the submitted weights | GPU-hours |
|---|---|---|
| MAST v3 network (`scripts/train_mast.sh`, 40 000 steps, global batch 256, bf16) | 8 × NVIDIA H100 80 GB, 1.71 h | 13.7 H100 GPU-h |
| DIII-D SimpleMLP + RidgeCV (`scripts/train_diiid.sh`, 50 epochs) | 1 × NVIDIA H100 80 GB, 6 min | 0.1 H100 GPU-h |
| **Total** | | **13.8 H100 GPU-h** |

Retraining with this package on 2026-10-31 took 1.82 h on 8 × H100 for MAST and 400 s on one H100 for
DIII-D (`docs/verification.md`).

**Single 80 GB A100-class GPU.** `scripts/train_mast.sh` with the default `NPROC=1` trains the same
model with the full global batch of 256 on one GPU. Measured on one H100 80 GB:
a 600-step run (2026-10-31) ran without running out of memory and sustained a median 957 windows/s (827–980), so 40 000 steps × 256 windows take about 3.0 h, 3.2 h with the periodic validation. Projected to one A100 80 GB, assuming bf16 throughput 2–3× lower than an H100:
about 6–10 h. Including DIII-D, the total stays within the 24 GPU-hour limit: at most about 10 GPU-hours on one A100.

## CPU-only data generation (not capped; declared)

| Step | Hardware | Wall-clock |
|---|---|---|
| FAIR-MAST download | public archive| — |
| FAIR-MAST cache build (`data_prep.scan`, `build_cache`, `stats`; 11 573 shots) | one CPU node, 20–55 cores | about 2 h |
| Sensor health audit (`data_prep.health features` + `run`) | one CPU node, ≤ 20 cores | about 1 h |
| DIII-D feature cache (`diiid/scripts/build_diiid_cache.py`, 7 041 shots) | 8 threads on a 20-core node | 8.4 min |

## Inference (for reference, not part of the training budget)

* MAST: v3 network on 8 × H100 for 1 205 shots, about 10 min; GSFit is CPU-only (dof=1, dof=2 and the
  passive solve).
* DIII-D: 879 shots, a few minutes on one GPU.