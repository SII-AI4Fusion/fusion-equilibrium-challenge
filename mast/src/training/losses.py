"""Masked reconstruction loss for the multimodal MAE."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from model import MAEOutput
from model.router import ModalitySpec, expand_patches


@dataclass
class LossTerms:
    total: Tensor
    per_modality: dict[str, Tensor]  # detached; only modalities with scored values


def reconstruction_losses(output: MAEOutput, targets: dict[str, Tensor],
                          masks: dict[str, Tensor], specs: list[ModalitySpec],
                          scored: Tensor) -> dict[str, Tensor]:
    """Mean squared error over observed values in ``scored`` ``[B, M, P]`` tokens, per modality.

    Each modality is normalized by its own number of scored values, so small and
    large modalities contribute on the same scale. Modalities without any scored
    value in the batch are omitted.
    """
    losses = {}
    for i, spec in enumerate(specs):
        if not scored[:, i].any():
            continue
        weight = expand_patches(scored[:, i].float(), spec) * masks[spec.name].float()
        denominator = weight.sum()
        if denominator <= 0:
            continue
        error = (output.reconstruction[spec.name].float() - targets[spec.name]).square()
        losses[spec.name] = (error * weight).sum() / denominator
    return losses


def compute_loss(output: MAEOutput, targets: dict[str, Tensor], masks: dict[str, Tensor],
                 specs: list[ModalitySpec], scored: Tensor,
                 modality_weights: dict[str, float] | None = None) -> LossTerms:
    """Weighted mean of the per-modality losses on ``scored`` tokens.

    Training scores the hidden tokens; plain reconstruction scores all available ones.
    """
    per_modality = reconstruction_losses(output, targets, masks, specs, scored)
    weights = modality_weights or {}
    # Every decoder stays in the graph even when its modality has no loss in this
    # batch, so distributed training never sees unused parameters.
    total = sum(r.sum() for r in output.reconstruction.values()) * 0.0
    total_weight = sum(weights.get(name, 1.0) for name in per_modality)
    if total_weight > 0:
        total = total + sum(weights.get(name, 1.0) * loss
                            for name, loss in per_modality.items()) / total_weight
    return LossTerms(total, {k: v.detach() for k, v in per_modality.items()})
