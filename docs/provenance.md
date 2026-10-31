# Data provenance

## DIII-D

* **Training data**: official challenge release v1.1.0, `Sophelio/fusion-equilibrium-challenge`
  revision `1e280905b85f2a6fdde7e06fca8cf3a1edf447cb`, split `diii_d_train` (7,041 shots), with the
  official `data_fixes.fix_d3d_ip_times` correction. Inputs: the 21 coil/Ip traces (`ECOILA`, `bcoil`,
  `F1A`…`F9B`, `plasma_current`) interpolated onto `efit_times`. Targets: `efit_psirz`, `efit_q95`,
  `efit_beta_n` of the same split.
* **Split**: by shot, `RandomState(42)` permutation, 80 % train / 10 % validation / 10 % held out.
  Validation loss only drives the learning-rate schedule; the last epoch is kept.
* **Test inputs**: Final release `ganatma/fusion-equilibrium-challenge-final` revision
  `965040c12b814f33c8e7209f6b1799932247e793`, `diii_d_private_test` (879 shots), inputs only.
* **No other data.** No external datasets, no pre-trained models, no public-test labels, no
  leaderboard feedback in training or model choice (architecture and hyperparameters are the starter
  kit's defaults).

## MAST

* **Test inputs**: `ganatma/fusion-equilibrium-challenge-final` revision
  `965040c12b814f33c8e7209f6b1799932247e793`, `mast_private_test` (1,205 shots): plasma current,
  PF/SOL and TF coil currents, coil geometry and Thomson profiles. No label column is read.
* **External data — FAIR-MAST** (public MAST archive, level 2 v3, CC BY-SA 4.0,
  `s3://mast/level2/shots`, see `data/download_fair_mast.py`): 11,573 shots, groups `summary`,
  `pf_active`, `tf`, `thomson_scattering`, `magnetics`, `pf_passive`. This trains the v3 network:
  inputs are Ip, PF/SOL and TF currents and Thomson n_e/T_e/p_e; targets are the measured slow
  magnetics (flux loops, pickup coils) and passive-structure currents. **No equilibrium data**:
  the `equilibrium` group (EFIT++) is not downloaded, cached, or used for training, validation,
  calibration or selection of any MAST component.
  * Split: by shot, a hash of the shot number (seed `crossmak-mast-v1`, 5 % validation, 5 % test);
    the validation shots pick the best checkpoint (lowest validation loss).
  * Sensor health audit (`data_prep.health`): flags dead, sign-flipped, mis-scaled, stuck or
    inconsistent channels from the same FAIR-MAST inputs and magnetics; flagged values are treated as
    missing during training.
* **Machine description**: `mast/assets/mast_static_geometry.npz` holds sensor and passive-structure
  positions, channel labels and the limiter, read from the FAIR-MAST shot 26497 store (`magnetics`,
  `pf_passive`, `wall` geometry arrays — static machine data, no measurements).
  `mast_static_geometry_passive.npz` is the copy the passive-current solve reads (same source).
  `mast/assets/output_grid.npz` contains only the 65 × 65 output-grid coordinates, taken from
  `efit_grid_R` and `efit_grid_Z` in the public Final inputs. The source revision, file and checksum
  are recorded in `mast/assets/output_grid.json`. No scoring-envelope mask is used by this revision.
  The original submission used `mask_coarse` from the official scoring envelope to compute the mean
  passive correction and restrict its application to the masked region.
* **Pre-trained models**: none. The v3 network is trained from random initialisation.