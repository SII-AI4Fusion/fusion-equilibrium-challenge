"""Passive-current correction, interpolated between converged keyframes.

`delta_all` (default, uses no competition asset): subtract the full-grid mean of psi_passive - psi_base and apply the
correction at every grid point.
`delta_mask` (reproduces submission 1012261): mean and correction restricted to `mask_coarse` of the starter
kit's MAST scoring envelope. Rule 3 mentions that file only for the official scoring, so this use may be
problematic; the variant is kept to reproduce the submission (see README).
q95 receives its keyframe difference in both; betaN is unchanged here (the inference entrypoint replaces it
with the dof=1 prediction).
"""

from __future__ import annotations
import numpy as np

VARIANTS = ("delta_all", "delta_mask")


def interp_keyframes(times, keys, values):
    """把关键帧上的值 `values[K, ...]` 按时间线性插值到全部 `times`，两端外推取常数。"""
    times = np.asarray(times, dtype=float)
    keys = np.asarray(keys, dtype=int)
    values = np.asarray(values, dtype=np.float64)
    if len(keys) != len(values) or len(keys) == 0:
        raise ValueError("Keyframe values must match a non-empty keyframe list")
    flat = values.reshape(len(keys), -1)
    kt = times[keys]
    if len(keys) == 1:
        return np.broadcast_to(flat[0], (len(times), flat.shape[1])).reshape(
            (len(times),) + values.shape[1:]
        ).copy()
    right = np.clip(np.searchsorted(kt, times, side="right"), 1, len(kt) - 1)
    left = right - 1
    w = np.clip((times - kt[left]) / (kt[right] - kt[left]), 0.0, 1.0)[:, None]
    out = (1.0 - w) * flat[left] + w * flat[right]
    return out.reshape((len(times),) + values.shape[1:])


def flux_delta(base_psi, passive_psi, keys, times, mask=None):
    """Subtract the mean correction (over the grid, or over `mask`) and interpolate to every frame;
    with `mask` the correction is zero outside it."""
    d = np.asarray(passive_psi, np.float64)[keys] - np.asarray(base_psi, np.float64)[keys]
    region = np.ones(d.shape[1:], dtype=bool) if mask is None else np.asarray(mask, bool)
    d = d - d[:, region].mean(axis=1)[:, None, None]
    if mask is not None:
        d = np.where(region, d, 0.0)
    return interp_keyframes(times, keys, d)


def scalar_delta(base, passive, keys, times):
    """标量关键帧差值插值；passive 在该关键帧无效时不作为该标量的关键帧。"""
    keys = np.asarray(keys, dtype=int)
    good = keys[np.isfinite(np.asarray(passive, float)[keys])]
    if good.size == 0:
        return np.zeros(len(times))
    return interp_keyframes(times, good, np.asarray(passive, float)[good] - np.asarray(base, float)[good])


def build_shot(base, passive, mask=None):
    """返回 {变体: {"psirz", "q95", "betaN"}}。

    base：原版提交中该炮的 psirz/q95/betaN（全部帧，有限值）。
    passive：`shots_gsfit/<i>.npz` 的内容（efit_ms、converged、psi_mono、q95、betaN_ts）。
    mask：给定时额外返回 `delta_mask`（starter kit envelope 的 mask_coarse，仅用于复现原提交）。
    """
    names = VARIANTS if mask is not None else ("delta_all",)
    times = np.asarray(passive["efit_ms"], dtype=float)
    keys = np.flatnonzero(np.asarray(passive["converged"], dtype=bool))
    base_psi = np.asarray(base["psirz"], np.float64)
    base_q = np.asarray(base["q95"], np.float64)
    base_b = np.asarray(base["betaN"], np.float64)
    if keys.size == 0:
        # passive 整炮未收敛：退回原版，避免引入非有限值。
        same = dict(psirz=base_psi.astype(np.float32), q95=base_q, betaN=base_b)
        return {name: dict(same) for name in names}, False
    q_delta = base_q + scalar_delta(base_q, passive["q95"], keys, times)
    out = dict(
        delta_all=dict(
            psirz=(base_psi + flux_delta(base_psi, passive["psi_mono"], keys, times)).astype(np.float32),
            q95=q_delta,
            betaN=base_b,
        ),
    )
    if mask is not None:
        out["delta_mask"] = dict(
            psirz=(base_psi + flux_delta(base_psi, passive["psi_mono"], keys, times, mask)).astype(np.float32),
            q95=q_delta,
            betaN=base_b,
        )
    return out, True
