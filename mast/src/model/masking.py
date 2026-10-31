"""Token masks for masked-autoencoder pretraining over (modality, patch) tokens.

A token is *available* when its patch holds at least one observed value. Masks
only ever hide available tokens, and every sample with at least two available
tokens keeps one visible. Sampling is vectorized and runs on the device of
``available``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


STRATEGIES = ("random", "modality", "time", "future")


@dataclass(frozen=True)
class MaskConfig:
    """``ratio`` is the target fraction to hide per sample.

    Each sample picks one strategy with probability proportional to its weight:

    - ``random``: independent tokens, ``ratio`` of the available tokens (standard MAE).
    - ``modality``: whole modalities in random order until at least ``ratio`` of the
      available tokens are hidden, so the realized fraction can exceed ``ratio``
      (cross-diagnostic prediction).
    - ``time``: one contiguous span of patches in every modality, covering ``ratio``
      of the patches that contain any available token.
    - ``future``: like ``time`` but the span ends at the last such patch (forecasting).
    """

    ratio: float = 0.0
    strategies: dict[str, float] = field(default_factory=lambda: {"random": 1.0})

    def __post_init__(self) -> None:
        if not 0 <= self.ratio < 1:
            raise ValueError("mask ratio must be in [0, 1)")
        unknown = set(self.strategies) - set(STRATEGIES)
        if unknown or not self.strategies or min(self.strategies.values()) < 0 \
                or sum(self.strategies.values()) <= 0:
            raise ValueError(f"invalid mask strategies: {self.strategies}")


