# Trained weights used for submission 1012261

The weight files are too large for git. They are published in the public Hugging Face model repo
[`shenmaa/fec26-sii-fusion-weights`](https://huggingface.co/shenmaa/fec26-sii-fusion-weights) (revision `ea109c04b12ca643a5c3d8469054f8ede44fbf55`)
and listed here with their checksums. Put them in `weights/mast_v3/` and `weights/diiid/`.

| File | Bytes | sha256 | Produced by |
|---|---|---|---|
| `mast_v3/best.pt` | 1027007886 | `892bd5621abb559c5635038ea8c6edd10bed6ad567a61de1f2e32419d915e69e` | 8 × H100, 1.71 h; checkpoint of step 37 000 of 40 000 (lowest validation loss) |
| `mast_v3/layout.json` | 115963 | `a6dbae9820deedb4b2de4b5a12b8ff0f774c451aceb75e7727a4e941e4335674` | training cache: channel layout |
| `mast_v3/stats.json` | 157623 | `684fe16ea20d5394658ef00926ab245103c33529372812a2bad467e0b4bf0e14` | training cache: normalisation statistics |
| `diiid/model.pt` | 19992372 | `5cc6aede238a0d5997a32a60a27ec047cbbb6a978e6dd354b0012dc418aa0547` | 1 × H100, 50 epochs |

`best.pt` holds the network weights and the training configuration. Inference also reads
`layout.json` and `stats.json` from the same directory (`--cache-dir`). `model.pt` holds the MLP
weights, the input standardisation and the two ridge regressions.

These files were trained with the original research code (crossmak commit `cdeb6df` and the starter-kit
script), whose training path is exactly what `scripts/train_mast.sh` (`NPROC=8`) and `scripts/train_diiid.sh`
run; see `docs/verification.md`.
