# Verification (2026-10-31)

All runs used this repository's entrypoints with the environments in `envs/`. GPU work ran on
NVIDIA H100 80 GB, and GSFit ran on the CPUs. GSFit was built from `third_party/gsfit`.

Sections 1 and 2 ran with `PASSIVE_VARIANT=delta_mask`, the variant the submission used (see the
README). Section 3 compares it with the default `delta_all`.

## 1. Shipped weights regenerate the submitted predictions

`scripts/infer_diiid.sh` and `scripts/infer_mast.sh` (4 nodes, `SHARDS=4`) on the Final inputs, compared
array by array with the two submitted files.

| Machine | Result |
|---|---|
| DIII-D | Same 2 637 arrays. q95 and βN identical in all 879 shots. ψ (float16) identical except for one-ulp roundings in 527 shots: max 4.9e-4, RMS 1.9e-6, 1 − R²(regenerated vs submitted) = 9.0e-11. |
| MAST | **Bit-identical**: all 3 615 arrays (ψ, q95, βN of 1 205 shots, 68 373 frames) equal the submitted file exactly. Wall-clock: prepare 14 min (CPU), v3 network 10 min (8 × H100), GSFit about 20 min per node on 4 nodes (160 CPUs each). |

The DIII-D differences are float16 roundings of a GPU forward pass run on a different node and
torch build than the original. The MAST network runs bf16 autocast on GPU, and GSFit is a nonlinear
iterative solver. The MAST pipeline reproduced exactly on H100; on another GPU generation small
differences should be expected.

## 2. Retraining from scratch

Both models were retrained from random initialisation with the package entrypoints. No equilibrium
reference was used in any comparison below. The Final labels are hidden, so the retrained predictions
are compared with the submitted predictions.

| Machine | Training run | Held-out / agreement |
|---|---|---|
| DIII-D | `scripts/train_diiid.sh` from the raw `diii_d_train` parquet (cache rebuilt: 7 041 shots, 1 559 340 frames), 1 × H100, 400 s | Held-out training shots: ψ R² 0.999533 (submitted model 0.999514), q95 R² 0.4106 and βN R² 0.5750, both identical to the submitted model. Final: q95 and βN identical in all 879 shots; ψ R²(retrained vs submitted) = 0.99982. |
| MAST | `NPROC=8 scripts/train_mast.sh`, 40 000 steps, 8 × H100, 1.82 h; best validation loss 0.001566 (submitted model 0.001581) | Final, `scripts/infer_mast.sh` with the retrained weights: all 68 373 frames finite; agreement with the submitted predictions R²(ψ) = 0.99982 (RMS 6.4e-4), R²(q95) = 0.994, R²(βN) = 0.995. |


The retrained predictions differ from the submitted ones far less than either differs from an
equilibrium reconstruction. We expect the composite S of a retrained model to be well within the 0.01
tolerance, but we cannot compute S here because the Final references are hidden.

## 3. The two passive-correction variants

The same saved GSFit solves as in sections 1 and 2 were packed with both variants (`finish` step only).
Each result is compared with the submitted MAST file.

| Weights | `PASSIVE_VARIANT` | ψ vs submitted | q95, βN vs submitted |
|---|---|---|---|
| shipped | `delta_mask` | bit-identical (1 205 / 1 205 shots) | bit-identical |
| shipped | `delta_all` | R² 0.99468, RMS 3.5e-3, max 0.045 | bit-identical |
| retrained | `delta_mask` | R² 0.99982, RMS 6.4e-4 | R² 0.994 / 0.995 |
| retrained | `delta_all` | R² 0.99451, RMS 3.6e-3 | R² 0.994 / 0.995 |

The two variants differ only in ψ, and only through the passive correction region; q95 and βN do not
depend on it. The `delta_all` predictions of the shipped weights are published at
[`shenmaa/fec26-sii-fusion-final-delta-all`](https://huggingface.co/datasets/shenmaa/fec26-sii-fusion-final-delta-all)
(revision `c98c773975d4621538ad769d4d77881d75be4b23`; `mast_private_test.npz` sha256
`6cb1f491286d2512226a664691bbc9791d2499b9ec70b01379934d35b758bca9`; `diii_d_private_test.npz` is the
submitted DIII-D file, sha256 `38778111269c80818b57b912f4c5b155d36a7de36aebc50a43ba84ffe76d26af`).
