# FusionEC-26 Challenge 2 reproduction package — team SII-Fusion

Team SII-Fusion is registered on Codabench as team **jygao**.

Reproduction package for the Final entry **Codabench submission 1012261** (scored 2026-10-27).
The submission had two prediction files:

| Machine | File | sha256 of the submitted file |
|---|---|---|
| DIII-D | `diii_d_private_test.npz` | `38778111269c80818b57b912f4c5b155d36a7de36aebc50a43ba84ffe76d26af` |
| MAST | `mast_private_test.npz` | `4d0e99f31e1b0bef6fecadccc4c804b2a82d9ed5a0931b8b9abf6813eeccce0b` |

* **DIII-D** — the starter-kit SimpleMLP baseline at full scale: the 21 coil/Ip traces
  (official `data_fixes.fix_d3d_ip_times` applied) → MLP 256-512-1024-4225 for ψ, RidgeCV for q95 and βN.
* **MAST** — a supervised transformer (the *v3 network*, 85 M
  parameters) is trained on the public FAIR-MAST archive to predict the 92 slow magnetic signals and 90
  passive-structure currents from Ip, PF/SOL coil currents, TF current and Thomson profiles — the
  inputs the challenge provides. GSFit (Tokamak Energy's equilibrium reconstruction code, vendored in
  `third_party/gsfit`) then reconstructs each frame from those predicted diagnostics.

## Two MAST passive-correction variants

The last MAST step adds a passive-current correction Δψ, computed on every 8th frame and interpolated
in time. `scripts/infer_mast.sh` offers two versions, selected with `PASSIVE_VARIANT`:

| `PASSIVE_VARIANT` | Correction | Status |
|---|---|---|
| `delta_all` (default) | mean of Δψ removed over the full 65 × 65 grid, correction applied everywhere | rule-compliant; uses no competition asset |
| `delta_mask` | mean removed and correction applied only inside `mask_coarse` of the starter kit's `fusion_scoring/masks/mast_envelope.npz` | **what submission 1012261 used**; reproduces it bit for bit |

Rule 3 allows `mast_envelope.npz` only inside the official scoring derivation. Using it in
`delta_mask` therefore does not comply with the rules. We found this while preparing this package. Both
variants are kept so the submitted predictions can be rebuilt exactly (`delta_mask`) and a compliant
pipeline is available (`delta_all`). The two differ only in this step: q95 and βN are identical, and ψ
differs as reported in [`docs/verification.md`](docs/verification.md#3-the-two-passive-correction-variants).

Data sources and their use: [`docs/provenance.md`](docs/provenance.md).
Training compute declaration: [`docs/compute.md`](docs/compute.md).
Methods report draft: [`docs/methods_report.md`](docs/methods_report.md).

## Layout

```
data/download_challenge.py   official challenge datasets (Hugging Face, pinned revisions, hash-checked)
data/download_fair_mast.py   FAIR-MAST level-2 Zarr shots (only the groups we read; never `equilibrium`)
scripts/                     entrypoints (below); env.sh selects the three interpreters
diiid/                       DIII-D feature cache, training and prediction
mast/src/                    MAST data cache, health audit, v3 network, training, inference, GSFit drivers
mast/frozen_v4/              frozen GSFit integration the MAST solves run through
mast/configs/                data, training and passive-solve configs
mast/assets/                 static machine geometry and public output-grid coordinates (see provenance)
third_party/gsfit/           GSFit 0.0.10 + fixes, commit 199d282 of SII-AI4Fusion/gsfit (BSD-3-Clause)
envs/                        pinned environments (requirements.lock per interpreter)
weights/                     manifest and checksums of the trained weights (files are a release asset)
```

## Environments

Three interpreters, each pinned exactly in `envs/<name>/requirements.lock` (the `# Python` line gives
the interpreter version). Create them with `uv` and point `scripts/env.sh` at them (defaults:
`.venv-torch`, `.venv-gsfit`, `.venv-data` in the repository root):

```bash
uv venv -p 3.10.12 .venv-torch && uv pip install -p .venv-torch -r envs/torch/requirements.lock    # torch 2.14 / CUDA 13
uv venv -p 3.13.7  .venv-gsfit && uv pip install -p .venv-gsfit -r envs/gsfit/requirements.lock
uv venv -p 3.11.11 .venv-data  && uv pip install -p .venv-data  -r envs/data/requirements.lock
# GSFit itself, from the vendored source (Rust 1.99, maturin)
uv pip install -p .venv-gsfit maturin && (cd third_party/gsfit && ../../.venv-gsfit/bin/maturin develop --release)
```

GPU inference uses bf16 autocast, so the MAST predictions depend at the 1e-4 level on the GPU
generation; run it on GPU (`DEVICE=cuda`, the default). GSFit runs on CPU: the 1 205 Final shots took
about 20 min per node on 4 nodes of 160 cores (`SHARDS=4`, `RANK=0..3`, shared run directory), so
about 1.5 h on one such node.

## Reproduce

```bash
# 0. data (direct downloads, hash-checked)
python data/download_challenge.py --which train --dest data/challenge-v1.1.0 --include data/diii_d_train/
python data/download_challenge.py --which final --dest data/challenge-final
bash scripts/prepare_mast_data.sh              # FAIR-MAST download -> cache -> sensor health audit (CPU)

# 1. train from scratch
bash scripts/train_diiid.sh                    # DIII-D, 1 GPU, ~6 min + CPU cache build
bash scripts/train_mast.sh                     # MAST v3 network, 1 GPU (NPROC=8 for 8 GPUs)

# 2. predict the Final inputs with the shipped weights (or WEIGHTS=<new run> after step 1)
uvx --from huggingface_hub hf download shenmaa/fec26-sii-fusion-weights --revision ea109c04b12ca643a5c3d8469054f8ede44fbf55 --local-dir weights
sha256sum -c weights/SHA256SUMS
bash scripts/infer_diiid.sh                    # -> runs/diiid_infer/diii_d_private_test.npz
bash scripts/infer_mast.sh all                 # -> runs/mast_infer/mast_private_test.npz (delta_all)
PASSIVE_VARIANT=delta_mask bash scripts/infer_mast.sh all   # rebuilds submission 1012261 exactly
```

`LIMIT_SHOTS=3` on the inference scripts and `SHOTS="26497 27000"` on `prepare_mast_data.sh` run small
smoke tests.

## Licence

Our code: MIT ([`LICENSE`](LICENSE)). `third_party/gsfit` keeps its BSD-3-Clause licence. The DIII-D
scripts and `diiid/data_fixes.py` derive from the MIT-licensed official starter kit
(Sophelio/fusion-equilibrium-challenge-starter); `mast/assets/output_grid.npz` contains only the
public Final input grid coordinates.
