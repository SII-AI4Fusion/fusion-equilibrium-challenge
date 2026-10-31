"""Per-kind convolutional encoders and decoders that map modalities to patch tokens.

Every codec turns one modality window into ``[batch, patches, token_dim]`` and
back from ``[batch, patches, decoder_dim]`` (``token_dim`` unless given), so
tokens of all modalities are aligned on the same time patches:

- ``TimeSeriesCodec``: ``[B, C, L]`` multichannel time series, 1D ResNet that
  halves time ``log2(samples_per_patch)`` times.
- ``ProfileCodec``: ``[B, C, R, L]`` radial profiles over time, 2D ResNet over
  (radius, time); the remaining radial cells are flattened into the token.
- ``FieldCodec``: ``[B, C, H, W, L]`` 2D fields, 2D ResNet on the frames of
  each patch separately.

Encoders receive the observation mask as extra input channels, so missing data
is distinguishable from a true zero.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ResBlock(nn.Module):
    def __init__(self, channels: int, dims: int) -> None:
        super().__init__()
        conv = nn.Conv1d if dims == 1 else nn.Conv2d
        self.block = nn.Sequential(
            nn.GroupNorm(_groups(channels), channels), nn.SiLU(),
            conv(channels, channels, 3, padding=1),
            nn.GroupNorm(_groups(channels), channels), nn.SiLU(),
            conv(channels, channels, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


def _widths(levels: int, base: int, maximum: int) -> list[int]:
    return [min(base * 2 ** i, maximum) for i in range(levels + 1)]


def halving_levels(size: int, max_levels: int, min_size: int = 1) -> int:
    """How many times ``size`` can be halved exactly while staying >= min_size."""
    levels = 0
    while levels < max_levels and size % 2 == 0 and size // 2 >= min_size:
        size //= 2
        levels += 1
    return levels


class _ConvStack(nn.Module):
    """Down- or up-sampling ResNet with an independent stride per axis and level."""

    def __init__(self, dims: int, in_channels: int, out_channels: int,
                 strides: list[tuple[int, ...]], widths: list[int], blocks: int,
                 upsample: bool) -> None:
        super().__init__()
        conv = nn.Conv1d if dims == 1 else nn.Conv2d
        self.upsample = upsample
        self.stem = conv(in_channels, widths[0], 3, padding=1)
        self.levels = nn.ModuleList()
        self.scales = [s for s in strides]
        for i, stride in enumerate(strides):
            layers = [ResBlock(widths[i], dims) for _ in range(blocks)]
            if upsample:
                layers.append(conv(widths[i], widths[i + 1], 3, padding=1))
            else:
                kernel = tuple(3 if s == 1 else 4 for s in stride)
                padding = tuple(1 for _ in stride)
                layers.append(conv(widths[i], widths[i + 1], kernel, stride=stride, padding=padding))
            self.levels.append(nn.Sequential(*layers))
        self.head = nn.Sequential(
            *[ResBlock(widths[-1], dims) for _ in range(blocks)],
            nn.GroupNorm(_groups(widths[-1]), widths[-1]), nn.SiLU(),
            conv(widths[-1], out_channels, 3 if upsample else 1, padding=1 if upsample else 0),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        for scale, level in zip(self.scales, self.levels):
            if self.upsample and any(s > 1 for s in scale):
                # Nearest upsampling + convolution avoids transposed-conv checkerboards.
                x = nn.functional.interpolate(x, scale_factor=scale, mode="nearest")
            x = level(x)
        return self.head(x)


class TimeSeriesCodec(nn.Module):
    def __init__(self, channels: int, samples_per_patch: int, token_dim: int,
                 base_width: int = 32, max_width: int = 256, blocks: int = 1,
                 decoder_dim: int | None = None) -> None:
        super().__init__()
        self.channels = channels
        self.samples_per_patch = samples_per_patch
        levels = int(math.log2(samples_per_patch))
        enc = _widths(levels, base_width, max_width)
        strides = [(2,)] * levels
        self.encoder = _ConvStack(1, 2 * channels, token_dim, strides, enc, blocks, upsample=False)
        dec = list(reversed(enc))
        self.decoder = _ConvStack(1, decoder_dim or token_dim, channels, strides, dec, blocks,
                                  upsample=True)

    def encode(self, x: Tensor, mask: Tensor) -> Tensor:
        tokens = self.encoder(torch.cat([x * mask, mask], dim=1))  # [B, D, P]
        return tokens.transpose(1, 2)

    def decode(self, tokens: Tensor) -> Tensor:
        return self.decoder(tokens.transpose(1, 2))  # [B, C, L]


class ProfileCodec(nn.Module):
    def __init__(self, channels: int, radius: int, samples_per_patch: int, token_dim: int,
                 base_width: int = 32, max_width: int = 256, blocks: int = 1,
                 max_radial_levels: int = 4, decoder_dim: int | None = None) -> None:
        super().__init__()
        self.channels = channels
        time_levels = int(math.log2(samples_per_patch))
        radial_levels = halving_levels(radius, max_radial_levels, min_size=2)
        levels = max(time_levels, radial_levels)
        strides = [(2 if i < radial_levels else 1, 2 if i < time_levels else 1) for i in range(levels)]
        self.radius_out = radius // 2 ** radial_levels
        enc = _widths(levels, base_width, max_width)
        self.feature = enc[-1]
        self.encoder = _ConvStack(2, 2 * channels, self.feature, strides, enc, blocks, upsample=False)
        self.to_token = nn.Linear(self.feature * self.radius_out, token_dim)
        self.from_token = nn.Linear(decoder_dim or token_dim, self.feature * self.radius_out)
        self.decoder = _ConvStack(2, self.feature, channels, list(reversed(strides)),
                                  list(reversed(enc)), blocks, upsample=True)

    def encode(self, x: Tensor, mask: Tensor) -> Tensor:
        h = self.encoder(torch.cat([x * mask, mask], dim=1))  # [B, F, R', P]
        b, f, r, p = h.shape
        return self.to_token(h.permute(0, 3, 1, 2).reshape(b, p, f * r))

    def decode(self, tokens: Tensor) -> Tensor:
        b, p, _ = tokens.shape
        h = self.from_token(tokens).reshape(b, p, self.feature, self.radius_out)
        return self.decoder(h.permute(0, 2, 3, 1))  # [B, C, R, L]


