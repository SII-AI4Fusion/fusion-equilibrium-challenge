"""Explicit resampling of native clocks and spatial axes onto cache grids.

Raw TokaMark signals keep their own clocks (200 Hz to 10 MHz). The cache puts
each modality on a uniform grid of ``samples_per_patch`` samples per patch:

- A faster native clock is low-pass filtered on the native clock and then read
  at bin centres (``method="antialias"``, the default). The filter is a
  Kaiser-windowed sinc with its cut-off between the grid's pass band
  (``PASS_FRACTION`` of the grid rate) and its Nyquist frequency, so content
  above Nyquist is suppressed by about ``STOPBAND_DB`` instead of folding back
  into the band. ``method="mean"`` keeps the plain bin average, whose ~13 dB
  side lobes alias heavily on clocks much faster than the grid (e.g. 10 MHz
  OMAHA onto 51.2 kHz); it is only used where a per-bin mean is the point
  (plasma current per patch).
- A comparable or slower clock is linearly interpolated at bin centres, but only
  when a finite native sample lies within ``max_gap_factor`` native steps.

Filtering never bridges a clock gap and is normalized by the finite samples
under the kernel; grid points whose kernel is not mostly covered by finite data
(segment ends, NaN runs) are NaN. Grid cells without supporting data are NaN;
the loader turns NaN into a mask.
"""

from __future__ import annotations

import numpy as np

# Pass band edge and Nyquist as fractions of the grid rate; the transition band
# lies between them, so the cached band up to 0.4 x rate is kept intact.
PASS_FRACTION = 0.4
STOP_FRACTION = 0.5
STOPBAND_DB = 80.0
# Fraction of the |kernel| weight that must fall on finite samples. 0.9 drops
# about 4 grid points at each segment end (in-band error there <= ~2%); 0.999
# would drop ~20, i.e. 20 patches of a one-sample-per-patch modality.
MIN_SUPPORT = 0.9
# Above this length the FFT convolution runs block by block (overlap-add).
_FFT_BLOCK = 1 << 20


def _clean_clock(values: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=float)
    if values.shape[-1] != times.shape[0]:
        raise ValueError(f"Time axis length {values.shape[-1]} does not match clock {times.shape[0]}")
    keep = np.isfinite(times)
    if not keep.all():
        values, times = values[..., keep], times[keep]
    if times.size > 1 and np.any(np.diff(times) < 0):
        order = np.argsort(times, kind="stable")
        values, times = values[..., order], times[order]
    return values, times


def resample_time(values: np.ndarray, times: np.ndarray, start: float, step: float,
                  length: int, *, max_gap_factor: float = 1.5,
                  method: str = "antialias") -> np.ndarray:
    """Resample ``values[..., time]`` onto bins ``start + [k, k + 1) * step``.

    ``method`` only matters when the native clock is faster than the grid:
    ``"antialias"`` low-pass filters before sampling at bin centres, ``"mean"``
    averages the finite samples of each bin. Returns float32 with the same
    leading shape and ``length`` samples.
    """
    if method not in ("antialias", "mean"):
        raise ValueError(f"Unknown resampling method {method!r}")
    values = np.asarray(values, dtype=np.float64)
    values, times = _clean_clock(values, times)
    out_shape = (*values.shape[:-1], length)
    if times.size == 0:
        return np.full(out_shape, np.nan, dtype=np.float32)
    native_step = float(np.median(np.diff(times))) if times.size > 1 else step
    if native_step <= 0:
        raise ValueError("Clock contains no increasing samples")

    if native_step * 1.5 <= step:
        if method == "mean":
            return _bin_mean(values, times, start, step, length).astype(np.float32)
        flat = values.reshape(-1, values.shape[-1])
        result = _antialias(flat, times, start, step, length, native_step, max_gap_factor)
        return result.reshape(out_shape).astype(np.float32)

    centres = start + step * (np.arange(length) + 0.5)
    max_gap = max_gap_factor * native_step
    finite = np.isfinite(values)
    flat = values.reshape(-1, values.shape[-1])
    flat_finite = finite.reshape(flat.shape)
    result = np.full((flat.shape[0], length), np.nan)
    complete = flat_finite.all(axis=1)
    if complete.any():
        # One set of weights serves every fully finite row.
        result[complete] = _interpolate_rows(flat[complete], times, centres, max_gap)
    for row in np.flatnonzero(~complete):
        valid = flat_finite[row]
        if valid.any():
            result[row] = _interpolate_rows(flat[row:row + 1, valid], times[valid], centres, max_gap)[0]
    return result.reshape(out_shape).astype(np.float32)


def _bin_mean(values: np.ndarray, times: np.ndarray, start: float, step: float,
              length: int) -> np.ndarray:
    finite = np.isfinite(values)
    edges = start + step * np.arange(length + 1)
    index = np.searchsorted(times, edges, side="left")
    zeros = np.zeros((*values.shape[:-1], 1))
    total = np.concatenate([zeros, np.cumsum(np.where(finite, values, 0.0), axis=-1)], axis=-1)
    count = np.concatenate([zeros, np.cumsum(finite, axis=-1)], axis=-1)
    sums = total[..., index[1:]] - total[..., index[:-1]]
    counts = count[..., index[1:]] - count[..., index[:-1]]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def lowpass_kernel(native_step: float, step: float) -> np.ndarray:
    """Odd-length Kaiser-windowed sinc on the native clock, unit DC gain.

    The cut-off sits midway between ``PASS_FRACTION`` and ``STOP_FRACTION`` of
    the grid rate ``1 / step``; the Kaiser formulas size the kernel for a
    ``STOPBAND_DB`` stop band over that transition width.
    """
    ratio = native_step / step  # grid rate in cycles per native sample
    cutoff = 0.5 * (PASS_FRACTION + STOP_FRACTION) * ratio
    width = (STOP_FRACTION - PASS_FRACTION) * ratio
    beta = 0.1102 * (STOPBAND_DB - 8.7)
    taps = int(np.ceil((STOPBAND_DB - 7.95) / (2.285 * 2 * np.pi * width))) + 1
    taps += 1 - taps % 2
    n = np.arange(taps) - taps // 2
    kernel = 2 * cutoff * np.sinc(2 * cutoff * n) * np.kaiser(taps, beta)
    return kernel / kernel.sum()


def _convolve_same(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """``np.convolve(row, kernel, "same")`` for every row of ``x`` via FFT."""
    n, m = x.shape[-1], kernel.size
    offset = (m - 1) // 2
    full_length = n + m - 1
    if full_length <= _FFT_BLOCK:
        nfft = 1 << (full_length - 1).bit_length()
        full = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(kernel, nfft), nfft)
        return full[..., offset:offset + n]
    # Overlap-add keeps memory bounded for long, fast clocks (10 MHz OMAHA).
    nfft = 1 << max(_FFT_BLOCK, 4 * m - 1).bit_length()
    block = nfft - m + 1
    spectrum = np.fft.rfft(kernel, nfft)
    full = np.zeros((*x.shape[:-1], full_length))
    for first in range(0, n, block):
        piece = x[..., first:first + block]
        out = np.fft.irfft(np.fft.rfft(piece, nfft) * spectrum, nfft)
        stop = min(first + piece.shape[-1] + m - 1, full_length)
        full[..., first:stop] += out[..., :stop - first]
    return full[..., offset:offset + n]


def _antialias(flat: np.ndarray, times: np.ndarray, start: float, step: float, length: int,
               native_step: float, max_gap_factor: float) -> np.ndarray:
    """Low-pass ``flat[row, time]`` per clock segment and read it at bin centres."""
    kernel = lowpass_kernel(native_step, step)
    magnitude = np.abs(kernel) / np.abs(kernel).sum()
    centres = start + step * (np.arange(length) + 0.5)
    result = np.full((flat.shape[0], length), np.nan)
    # Clock gaps split the signal; the filter never reaches across them.
    breaks = np.flatnonzero(np.diff(times) > max_gap_factor * native_step) + 1
    bounds = np.concatenate([[0], breaks, [times.size]])
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        seg_times = times[lo:hi]
        inside = np.flatnonzero((centres >= seg_times[0]) & (centres <= seg_times[-1]))
        if inside.size == 0 or hi - lo < 2:
            continue
        right = np.clip(np.searchsorted(seg_times, centres[inside]), 1, seg_times.size - 1)
        left = right - 1
        span = seg_times[right] - seg_times[left]
        frac = np.clip((centres[inside] - seg_times[left]) / np.where(span > 0, span, 1), 0, 1)

        def at_centres(filtered: np.ndarray) -> np.ndarray:
            return filtered[..., left] * (1 - frac) + filtered[..., right] * frac

        segment = flat[:, lo:hi]
        finite = np.isfinite(segment)
        complete = finite.all(axis=1)
        ones = np.ones((1, hi - lo))
        shared_weight = at_centres(_convolve_same(ones, kernel))[0]
        shared_cover = at_centres(_convolve_same(ones, magnitude))[0]
        for row in range(flat.shape[0]):
            if complete[row]:
                weight, cover = shared_weight, shared_cover
                filtered = at_centres(_convolve_same(segment[row:row + 1], kernel))[0]
            elif finite[row].any():
                mask = finite[row:row + 1].astype(np.float64)
                weight = at_centres(_convolve_same(mask, kernel))[0]
                cover = at_centres(_convolve_same(mask, magnitude))[0]
                filtered = at_centres(_convolve_same(np.where(finite[row:row + 1],
                                                              segment[row:row + 1], 0.0), kernel))[0]
            else:
                continue
            good = (cover >= MIN_SUPPORT) & (weight > 0.5)
            with np.errstate(invalid="ignore", divide="ignore"):
                result[row, inside] = np.where(good, filtered / weight, np.nan)
    return result


def _interpolate_rows(rows: np.ndarray, times: np.ndarray, centres: np.ndarray,
                      max_gap: float) -> np.ndarray:
    if times.size == 1:
        out = np.repeat(rows, centres.size, axis=1)
        out[:, np.abs(centres - times[0]) > max_gap] = np.nan
        return out
    right = np.clip(np.searchsorted(times, centres), 1, times.size - 1)
    left = right - 1
    span = times[right] - times[left]
    weight = np.clip((centres - times[left]) / np.where(span > 0, span, 1), 0, 1)
    out = rows[:, left] * (1 - weight) + rows[:, right] * weight
    nearest = np.minimum(np.abs(centres - times[left]), np.abs(centres - times[right]))
    out[:, nearest > max_gap] = np.nan
    return out




def resize_axis(values: np.ndarray, axis: int, size: int) -> np.ndarray:
    """Linearly resample an index axis to ``size`` points, keeping both ends.

    Used for spatial axes whose length varies between shots (q profile, LCFS
    points) or is odd (the 65 x 65 equilibrium grid). NaN propagates to the
    neighbouring output points.
    """
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[axis]
    if n == size:
        return values
    if n == 1:
        return np.repeat(values, size, axis=axis)
    position = np.linspace(0, n - 1, size)
    left = np.clip(np.floor(position).astype(int), 0, n - 2)
    weight = position - left
    shape = [1] * values.ndim
    shape[axis] = size
    weight = weight.reshape(shape)
    return (np.take(values, left, axis=axis) * (1 - weight)
            + np.take(values, left + 1, axis=axis) * weight)


