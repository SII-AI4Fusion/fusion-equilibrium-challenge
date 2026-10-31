"""Multimodal masked autoencoder over (modality, time-patch) tokens.

    modality tensors ──router.encode (hidden patches zeroed)──> tokens [B, M, P, E]
        + modality / patch / absolute-time embeddings
        ──Transformer encoder over visible tokens only──> encoded [B, M, P, E]
        ──Linear(E → D); hidden positions get a learned mask token
        + decoder embeddings
        ──Transformer decoder over visible ∪ query tokens──router.decode──> reconstructions

A token is *available* when its patch holds at least one observed value; it is
*visible* when available and not hidden. The encoder sees only visible tokens,
so its cost scales with the visible count. *Query* tokens are the positions
whose reconstruction the caller wants; the decoder attends to visible and query
tokens, and reconstructions outside them are not meaningful. Training queries
every position, so availability of hidden tokens is never an input. Codecs only
run on the samples that need them (e.g. BES is observed in ~20% of windows).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any

import torch
from torch import Tensor, nn

from .router import CodecConfig, ModalityRouter, ModalitySpec


@dataclass(frozen=True)
class MAEConfig:
    encoder_dim: int = 256
    encoder_layers: int = 6
    encoder_heads: int = 8
    decoder_dim: int = 256
    decoder_layers: int = 4
    decoder_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    codec: CodecConfig = field(default_factory=CodecConfig)
    # Per-kind or per-name CodecConfig fields, e.g. {"field": {"base_width": 64}}.
    codec_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> MAEConfig:
        raw = dict(raw)
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise KeyError(f"Unknown model keys: {sorted(unknown)}")
        return cls(codec=CodecConfig(**raw.pop("codec", {})), **raw)


@dataclass
class MAEOutput:
    reconstruction: dict[str, Tensor]  # per modality, same shape as the input
    available: Tensor                  # [B, M, P] bool
    visible: Tensor                    # [B, M, P] bool, seen by the encoder
    hidden: Tensor                     # [B, M, P] bool, available but masked


def _transformer(layers: int, dim: int, heads: int, mlp_ratio: float, dropout: float) -> nn.Module:
    layer = nn.TransformerEncoderLayer(dim, heads, int(dim * mlp_ratio), dropout,
                                       activation="gelu", batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(dim), enable_nested_tensor=False)


class TimeEmbedding(nn.Module):
    """Fourier features of absolute time in seconds (window position within the shot)."""

    def __init__(self, dim: int, frequencies: int = 16) -> None:
        super().__init__()
        self.register_buffer("scales", 2 * math.pi * 2.0 ** torch.arange(frequencies) / 4.0,
                             persistent=False)
        self.project = nn.Linear(2 * frequencies, dim)

    def forward(self, seconds: Tensor) -> Tensor:
        angles = seconds[..., None] * self.scales
        return self.project(torch.cat([angles.sin(), angles.cos()], dim=-1))


class Positions(nn.Module):
    """Learned modality and patch embeddings plus Fourier features of absolute time."""

    def __init__(self, modalities: int, patches: int, dim: int, patch_seconds: float) -> None:
        super().__init__()
        self.patch_seconds = patch_seconds
        self.modality = nn.Parameter(torch.zeros(modalities, dim))
        self.patch = nn.Parameter(torch.zeros(patches, dim))
        self.time = TimeEmbedding(dim)
        nn.init.normal_(self.modality, std=0.02)
        nn.init.normal_(self.patch, std=0.02)

    def forward(self, t0: Tensor) -> Tensor:
        """``[B, M, P, dim]`` for window start times ``t0`` (seconds)."""
        patches = self.patch.shape[0]
        seconds = t0[:, None] + self.patch_seconds * (torch.arange(patches, device=t0.device) + 0.5)
        per_patch = self.patch[None] + self.time(seconds)           # [B, P, dim]
        return self.modality[None, :, None] + per_patch[:, None]    # [B, M, P, dim]


def run_on_subset(transformer: nn.Module, tokens: Tensor, keep: Tensor) -> Tensor:
    """Apply ``transformer`` to the kept tokens of every sample.

    ``tokens`` is ``[B, N, D]`` and ``keep`` ``[B, N]`` bool. Kept tokens are
    gathered to the front and padded to the largest kept count, so cost scales
    with that count rather than N. Returns ``[B, N, D]`` with zeros where
    ``keep`` is False; samples without any kept token get all zeros.
    """
    b, n, d = tokens.shape
    counts = keep.sum(dim=1)
    length = max(int(counts.max()), 1)
    order = (~keep).to(torch.int8).argsort(dim=1, stable=True)[:, :length]  # kept first
    rows = torch.arange(b, device=tokens.device)[:, None]
    padding = torch.arange(length, device=tokens.device)[None] >= counts[:, None]
    # A row with no kept token would attend to nothing (NaN); let it see one slot
    # and discard the result below.
    padding[counts == 0, 0] = False
    out = transformer(tokens[rows, order], src_key_padding_mask=padding)
    out = out.masked_fill(padding[..., None] | (counts == 0)[:, None, None], 0.0)
    result = out.new_zeros(b, n, out.shape[-1])
    result[rows, order] = out
    return result.masked_fill(~keep[..., None], 0.0)


class MultiModalMAE(nn.Module):
    def __init__(self, modalities: list[ModalitySpec | dict[str, Any]], patches: int,
                 patch_seconds: float, config: MAEConfig) -> None:
        super().__init__()
        specs = [m if isinstance(m, ModalitySpec) else ModalitySpec.from_dict(m) for m in modalities]
        self.config = config
        self.patches = patches
        self.patch_seconds = patch_seconds
        e, d, m = config.encoder_dim, config.decoder_dim, len(specs)
        self.router = ModalityRouter(specs, e, config.codec, config.codec_overrides, decoder_dim=d)
        self.encoder_positions = Positions(m, patches, e, patch_seconds)
        self.encoder = _transformer(config.encoder_layers, e, config.encoder_heads,
                                    config.mlp_ratio, config.dropout)
        self.encoder_to_decoder = nn.Linear(e, d)
        self.mask_token = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.mask_token, std=0.02)
        self.decoder_positions = Positions(m, patches, d, patch_seconds)
        self.decoder = _transformer(config.decoder_layers, d, config.decoder_heads,
                                    config.mlp_ratio, config.dropout)

    @property
    def specs(self) -> list[ModalitySpec]:
        return self.router.specs


    def availability(self, masks: dict[str, Tensor]) -> Tensor:
        """``[B, M, P]`` bool: the patch holds at least one observed value."""
        return self.router.patch_coverage(masks, self.patches) > 0

    def encode(self, inputs: dict[str, Tensor], masks: dict[str, Tensor], t0: Tensor,
               hidden: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """Return encoded tokens ``[B, M, P, E]`` (zero where not visible), available and visible."""
        available = self.availability(masks)
        visible = available if hidden is None else available & ~hidden
        b, m, p = available.shape
        tokens = self.router.encode(inputs, masks, ~visible)
        tokens = tokens + self.encoder_positions(t0)
        encoded = run_on_subset(self.encoder, tokens.reshape(b, m * p, -1), visible.reshape(b, -1))
        return encoded.reshape(b, m, p, -1), available, visible

    def decode(self, encoded: Tensor, visible: Tensor, queries: Tensor, t0: Tensor,
               wanted: Tensor | None = None) -> dict[str, Tensor]:
        """Reconstruct every modality from the visible ∪ ``queries`` decoder context.

        Only positions in ``wanted`` (default ``queries``) are meaningful; a sample
        without any wanted position of a modality skips its codec and is zero.
        """
        b, m, p = visible.shape
        tokens = torch.where(visible[..., None], self.encoder_to_decoder(encoded),
                             self.mask_token.to(encoded.dtype))
        tokens = tokens + self.decoder_positions(t0)
        attend = (visible | queries).reshape(b, -1)
        decoded = run_on_subset(self.decoder, tokens.reshape(b, m * p, -1), attend)
        wanted = queries if wanted is None else wanted
        return self.router.decode(decoded.reshape(b, m, p, -1), wanted.any(dim=2))

    def forward(self, inputs: dict[str, Tensor], masks: dict[str, Tensor], t0: Tensor,
                hidden: Tensor | None = None) -> MAEOutput:
        """Training pass: encode visible tokens, reconstruct every available token.

        The decoder queries every position, as ``reconstruct`` does by default, so
        it is never told which hidden positions hold data: availability carries
        physics (e.g. charge exchange and BES need a neutral beam).
        """
        encoded, available, visible = self.encode(inputs, masks, t0, hidden)
        reconstruction = self.decode(encoded, visible, torch.ones_like(available), t0, wanted=available)
        if self.training:
            # Keep the mask token in the graph when a batch happens to hide nothing,
            # so distributed training never sees an unused parameter.
            first = next(iter(reconstruction))
            reconstruction[first] = reconstruction[first] + 0.0 * self.mask_token.sum()
        return MAEOutput(reconstruction, available, visible, available & ~visible)

    def reconstruct(self, inputs: dict[str, Tensor], masks: dict[str, Tensor], t0: Tensor,
                    queries: Tensor | None = None) -> tuple[dict[str, Tensor], Tensor]:
        """Predict ``queries`` positions (default: all) from every observed value.

        Returns the reconstructions and visible ``[B, M, P]``. ``queries`` is a
        fixed choice of the caller and does not depend on which targets exist.
        """
        encoded, _, visible = self.encode(inputs, masks, t0)
        if queries is None:
            queries = torch.ones_like(visible)
        return self.decode(encoded, visible, queries, t0), visible

