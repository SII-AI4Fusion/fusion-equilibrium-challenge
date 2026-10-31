# Methods report — team SII-Fusion (Codabench: jygao), FusionEC-26 Challenge 2 (draft)

## Summary

The two machines use unrelated methods. **DIII-D** is the starter kit's SimpleMLP baseline trained at
full scale on the official training split. **MAST** has no equilibrium training data under the rules,
so we do not learn ψ at all: a network trained on public raw diagnostics predicts the magnetic
measurements the challenge inputs lack, and a physics equilibrium reconstruction (GSFit) turns them
into ψ, q95 and βN.

## DIII-D

Inputs are the 21 coil and plasma-current traces, with the official Ip time-axis fix, interpolated
onto the EFIT times and standardised. An MLP (256-512-1024-4225, ReLU, BatchNorm) maps them to the
65 × 65 ψ map (MSE, Adam 1e-3, batch 256, 50 epochs, ReduceLROnPlateau); q95 and βN come from RidgeCV
on the same features. Training uses `diii_d_train` only, split by shot (80/10/10). ψ is submitted as
float16.

## MAST

1. **Diagnostic predictor (v3 network).** A transformer encoder-decoder (85 M parameters; 6 + 4
   layers, width 256, per-modality convolutional patch codecs over 32 windows of 5 ms) takes Ip, PF/SOL
   and TF currents and Thomson profiles and predicts the 92 slow magnetic signals (flux loops and
   pickup coils) and 90 passive-structure currents. It is trained from scratch on the public FAIR-MAST
   archive (11,573 shots, raw diagnostics only; no EFIT++ data), with a sensor-health audit masking
   broken channels in the targets. 40,000 steps, global batch 256, AdamW 3e-4 with cosine decay, bf16.
2. **Equilibrium reconstruction.** For every Final frame GSFit (Tokamak Energy, BSD-3) fits the
   predicted magnetics plus the measured coil currents and Ip with polynomial p′ and FF′ profiles
   (2 degrees of freedom each, diagonal regularisation ×150, OBR02 pickup weight ×2, R grid
   0.06–2.0 m). Frames that do not converge (434 of 68,373) take the 1-degree-of-freedom solution.
3. **Passive-current correction.** On every 8th frame a second GSFit solve adds the
   predicted passive currents. Its difference from step 2, Δψ minus its full-grid mean, and the q95
   difference are interpolated in time between these key frames. They are then
   added to the step-2 solution over the full grid.
4. βN is taken from the 1-degree-of-freedom solve.

## Data use, external data and feedback

* External data: FAIR-MAST level-2 v3 (public, CC BY-SA 4.0), raw diagnostics only, and the static
  machine description from one FAIR-MAST shot. No pre-trained models.
* The FAIR-MAST training shots were not filtered against the challenge test sets. They include the
  shots that correspond to the public-test and Final MAST shots, whose measured magnetics and passive
  currents (not equilibria) the network therefore saw during training.
* Full details: `docs/provenance.md`.

## Compute

See `docs/compute.md`.

## Use of AI coding assistants

AI coding assistants (Anthropic's Claude Code and OpenAI's Codex) wrote and ran most of the code under
our direction: data preparation, the network and training code, the GSFit integration, job
submission, experiment bookkeeping, and the preparation of this reproduction package. We reviewed the
results and made the decisions, and we take responsibility for the submission.
