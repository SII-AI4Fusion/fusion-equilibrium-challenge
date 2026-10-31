"""GSFit 0.0.10 的 MAST v4 输入适配和稳定标量提取；不读取平衡参考。"""

from __future__ import annotations
import numpy as np
from training.gsfit_postprocess import E_CHARGE, MU0

PROTOCOL = dict(
    dof=1,
    regularisation=1.0,
    initial_r=0.9,
    minor_radius=0.35,
    kappa=1.5,
    max_iter=100,
    bp_sigma=0.002,
    flux_sigma=0.005,
    passives="omit",
    holdout_stride=0,
    exclude_sensor=["P4L/1", "P4L/4"],
    ts_pressure_weight=0.0,
    ts_pressure_alpha=2.0,
    ts_pressure_expected_pa=2000.0,
    ts_kinetic_fallback=False,
    passive_checkpoint="",
    passive_scale=1.0,
    passive_monopole_correct=False,
)


def geometry_indices(channels, geometry_channels, flux=False):
    names = [str(name).upper() for name in geometry_channels]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate geometry names")
    indices = {name: index for index, name in enumerate(names)}
    keys = [
        ("FL_" + str(name).replace("/", "_")) if flux else str(name)
        for name in channels
    ]
    missing = [name for name in keys if name.upper() not in indices]
    if missing:
        raise ValueError(f"Missing geometry: {missing}")
    return [indices[name.upper()] for name in keys]


def psi_to_weber_per_rad(psi):
    return np.asarray(psi) / (2 * np.pi)


def fit_settings(sigma, kind="bp"):
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Diagnostic uncertainty must be positive")
    if kind not in ("bp", "flux", "ip"):
        raise ValueError("Unknown diagnostic kind")
    multiplier = 2 * np.pi if kind == "flux" else 1.0
    return {
        "fit_settings_expected_value": 1.0,
        "fit_settings_weight": 1.0 / (multiplier * sigma),
    }


def sensor_role(index, name, stride, excluded):
    if name in excluded:
        return "excluded"
    return "heldout" if stride and index % stride == 0 else "fit"


def interp_thomson_electron_pressure(
    ts_t, te_profile, ne_profile, target_t: float, max_extrap_s: float = 0.015
) -> np.ndarray:
    ts_t = np.asarray(ts_t, dtype=float)
    te_profile = np.asarray(te_profile, dtype=float)
    ne_profile = np.asarray(ne_profile, dtype=float)
    if (
        ts_t.ndim != 1
        or te_profile.ndim != 2
        or ne_profile.shape != te_profile.shape
        or te_profile.shape[0] != ts_t.size
    ):
        return np.full(te_profile.shape[-1] if te_profile.ndim >= 1 else 0, np.nan)
    n_channels = te_profile.shape[1]
    pe = np.full(n_channels, np.nan, dtype=float)
    for k in range(n_channels):
        ok = np.isfinite(ts_t) & np.isfinite(te_profile[:, k]) & np.isfinite(ne_profile[:, k])
        if ok.sum() < 2:
            continue
        order = np.argsort(ts_t[ok])
        t_ok = ts_t[ok][order]
        if target_t < t_ok[0] - max_extrap_s or target_t > t_ok[-1] + max_extrap_s:
            continue
        t_clamped = float(np.clip(target_t, t_ok[0], t_ok[-1]))
        te_k = float(np.interp(t_clamped, t_ok, te_profile[ok, k][order]))
        ne_k = float(np.interp(t_clamped, t_ok, ne_profile[ok, k][order]))
        val = E_CHARGE * ne_k * te_k
        if np.isfinite(val) and val >= 0.0:
            pe[k] = val
    return pe


def dense_limiter(r, z, spacing=0.03):
    points = np.column_stack([r, z])
    if not np.array_equal(points[0], points[-1]):
        points = np.vstack([points, points[0]])
    dense = []
    for left, right in zip(points[:-1], points[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(right - left) / spacing)))
        dense.extend(left + (right - left) * np.arange(n)[:, None] / n)
    dense = np.asarray(dense)
    return dense[:, 0].copy(), dense[:, 1].copy()


def _ellipke_agm(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.clip(np.asarray(m, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    a = np.ones_like(m)
    b = np.sqrt(1.0 - m)
    c = np.sqrt(m)
    s = 0.5 * c * c
    twopow = 1.0
    for _ in range(7):
        c = 0.5 * (a - b)
        a_next = 0.5 * (a + b)
        b = np.sqrt(a * b)
        a = a_next
        twopow *= 2.0
        s = s + twopow * c * c
    k_val = np.pi / (2.0 * a)
    e_val = k_val * (1.0 - s)
    return k_val, e_val


def green_psi_per_rad(r, z, rc: float, zc: float) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    k2 = 4.0 * r * rc / ((r + rc) ** 2 + (z - zc) ** 2 + 1e-12)
    k2 = np.clip(k2, 1e-12, 1.0 - 1e-12)
    k = np.sqrt(k2)
    ek, ee = _ellipke_agm(k2)
    return (MU0 / (2.0 * np.pi)) * np.sqrt(r * rc) / k * ((2.0 - k2) * ek - 2.0 * ee)


