"""Route each modality to the encoder/decoder that matches its data kind."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import Tensor, nn

from .codecs import ProfileCodec, TimeSeriesCodec


@dataclass(frozen=True)
class ModalitySpec:
    """Tensor layout of one modality window: ``[B, channels, *spatial, patches * spp]``."""

    name: str
    kind: str
    channels: int
    samples_per_patch: int
    spatial: tuple[int, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ModalitySpec:
        return cls(raw["name"], raw["kind"], int(raw["channels"]),
                   int(raw["samples_per_patch"]), tuple(raw.get("spatial", ())))


@dataclass(frozen=True)
class CodecConfig:
    base_width: int = 32
    max_width: int = 256
    blocks: int = 1


def build_codec(spec: ModalitySpec, token_dim: int, config: CodecConfig,
                decoder_dim: int | None = None) -> nn.Module:
    """Codec encoding to ``token_dim`` and decoding from ``decoder_dim`` (default ``token_dim``)."""
    common = {"token_dim": token_dim, "base_width": config.base_width,
              "max_width": config.max_width, "blocks": config.blocks,
              "decoder_dim": decoder_dim}
    if spec.kind == "timeseries":
        return TimeSeriesCodec(spec.channels, spec.samples_per_patch, **common)
    if spec.kind == "profile":
        return ProfileCodec(spec.channels, spec.spatial[0], spec.samples_per_patch, **common)
    raise ValueError(f"No codec for modality kind {spec.kind!r}")


def expand_patches(patch_values: Tensor, spec: ModalitySpec) -> Tensor:
    """Broadcast ``[B, P]`` patch values to the modality's ``[B, 1, *spatial, L]``."""
    expanded = patch_values.repeat_interleave(spec.samples_per_patch, dim=1)
    return expanded.reshape(expanded.shape[0], 1, *([1] * len(spec.spatial)), expanded.shape[1])


def on_rows(function: Callable[..., Tensor], rows: Tensor, *values: Tensor) -> Tensor:
    """``function`` applied to the samples where ``rows`` ``[B]`` is True; zeros elsewhere.

    One sample is processed even when none is selected, so every codec stays in
    the autograd graph (distributed training needs a gradient for every parameter).
    """
    index = rows.nonzero().squeeze(1)
    if index.numel() == rows.shape[0]:
        return function(*values)
    if index.numel() == 0:
        index = index.new_zeros(1)
    result = function(*(value[index] for value in values))
    return result.new_zeros(rows.shape[0], *result.shape[1:]).index_copy(0, index, result)


class ModalityRouter(nn.Module):
    """Holds one codec per modality and converts between tensors and patch tokens.

    ``codec_overrides`` maps a modality kind or name to CodecConfig fields, e.g.
    ``{"field": {"base_width": 16}}``; a name takes precedence over a kind.
    """

    def __init__(self, specs: list[ModalitySpec], token_dim: int, config: CodecConfig,
                 codec_overrides: dict[str, dict[str, Any]] | None = None,
                 decoder_dim: int | None = None) -> None:
        super().__init__()
        overrides = codec_overrides or {}
        self.specs = list(specs)
        self.codecs = nn.ModuleDict()
        for spec in self.specs:
            fields = {**config.__dict__, **overrides.get(spec.kind, {}), **overrides.get(spec.name, {})}
            self.codecs[spec.name] = build_codec(spec, token_dim, CodecConfig(**fields), decoder_dim)


    def patch_coverage(self, masks: dict[str, Tensor], patches: int) -> Tensor:
        """Fraction of observed values in every (modality, patch): ``[B, M, P]``."""
        coverage = []
        for spec in self.specs:
            mask = masks[spec.name].float()
            b = mask.shape[0]
            # Time is last; group its samples by patch and average everything else.
            per_patch = mask.reshape(b, -1, patches, spec.samples_per_patch)
            coverage.append(per_patch.mean(dim=(1, 3)))
        return torch.stack(coverage, dim=1)

    def encode(self, inputs: dict[str, Tensor], masks: dict[str, Tensor],
               hidden: Tensor) -> Tensor:
        """Encode all modalities to ``[B, M, P, D]``; hidden patches are zeroed first,
        so their content cannot reach any token through convolutions.

        A sample whose modality is hidden or unobserved in every patch skips that
        codec and gets zero tokens; callers must not use tokens of hidden patches.
        """
        tokens = []
        for i, spec in enumerate(self.specs):
            keep = expand_patches((~hidden[:, i]).to(inputs[spec.name].dtype), spec)
            mask = masks[spec.name].to(inputs[spec.name].dtype) * keep
            codec = self.codecs[spec.name]
            tokens.append(on_rows(lambda x, m, codec=codec: codec.encode(x * m, m),
                                  (~hidden[:, i]).any(dim=1), inputs[spec.name], mask))
        return torch.stack(tokens, dim=1)

    def decode(self, tokens: Tensor, rows: Tensor | None = None) -> dict[str, Tensor]:
        """Decode ``[B, M, P, decoder_dim]`` tokens to one tensor per modality.

        With ``rows`` ``[B, M]`` only the samples where it is True are decoded for
        each modality; the others are zero.
        """
        if rows is None:
            rows = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        return {spec.name: on_rows(self.codecs[spec.name].decode, rows[:, i], tokens[:, i])
                for i, spec in enumerate(self.specs)}
