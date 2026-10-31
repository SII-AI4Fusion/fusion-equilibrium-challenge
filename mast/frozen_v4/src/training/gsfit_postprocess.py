"""从磁通、F/p 剖面稳定计算 GSFit 标量，不依赖原生边界点排序。"""
from __future__ import annotations

import numpy as np

MU0 = 4.0e-7 * np.pi


def _bilinear(field, r, z, points):
    ir = np.clip(np.searchsorted(r, points[:, 0], side="right") - 1, 0, len(r) - 2)
    iz = np.clip(np.searchsorted(z, points[:, 1], side="right") - 1, 0, len(z) - 2)
    ar = (points[:, 0] - r[ir]) / (r[ir + 1] - r[ir])
    az = (points[:, 1] - z[iz]) / (z[iz + 1] - z[iz])
    return ((1 - ar) * (1 - az) * field[iz, ir] + ar * (1 - az) * field[iz, ir + 1]
            + (1 - ar) * az * field[iz + 1, ir] + ar * az * field[iz + 1, ir + 1])


def _closed_contour(generator, level, axis, vessel=None):
    from matplotlib.path import Path

    candidates = []
    for points in generator.lines(float(level)):
        if len(points) < 5 or np.linalg.norm(points[0] - points[-1]) > 1e-8:
            continue
        if not np.isfinite(points).all() or not Path(points).contains_point(axis):
            continue
        if vessel is not None:
            # Vertex checks alone miss a segment crossing a concave wall.
            if (not vessel.contains_points(points).all()
                    or vessel.intersects_path(Path(points), filled=False)):
                continue
        r, z = points.T
        area = abs(np.sum(r[:-1] * z[1:] - r[1:] * z[:-1])) / 2
        if area > 0:
            candidates.append((area, points))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def _geometry(contour):
    r, z = contour.T
    cross = r[:-1] * z[1:] - r[1:] * z[:-1]
    volume = 2 * np.pi * abs(np.sum((r[:-1] + r[1:]) * cross) / 6)
    minor_radius = float(np.ptp(r) / 2)
    major_radius = float((r.min() + r.max()) / 2)
    return float(volume), minor_radius, major_radius


def stable_scalars(psi, r, z, psi_axis, psi_boundary, f_profile, p_profile,
                   ip, i_rod, axis_r, axis_z, vessel=None):
    """psi 和两端磁通为 Wb/rad；F 为 T*m、p 为 Pa、电流为 A。

    q95 用包含磁轴的闭合 95% 磁通线积分。βN 用稳定轮廓的体积和
    p(ψ) 体积分；分离面不闭合时从接近边界的闭合面求几何并记录层级。
    无法计算的量返回 NaN，交由同炮独立标量缺失策略处理。
    提供 vessel 时，只使用完全位于真空室内部的轮廓。
    """
    import contourpy
    from matplotlib.path import Path

    psi, r, z = np.asarray(psi, float), np.asarray(r, float), np.asarray(z, float)
    f_profile, p_profile = np.asarray(f_profile, float), np.asarray(p_profile, float)
    result = dict(q95=float("nan"), betaN=float("nan"), volume=float("nan"),
                  boundary_level=float("nan"), q95_error="", betaN_error="")
    span = float(psi_boundary - psi_axis)
    if (psi.shape != (z.size, r.size) or not np.isfinite(psi).all()
            or not np.isfinite(span) or abs(span) < 1e-15
            or np.any(np.diff(r) <= 0) or np.any(np.diff(z) <= 0)):
        result.update(q95_error="invalid flux/grid", betaN_error="invalid flux/grid")
        return result
    normalized = (psi - psi_axis) / span
    generator = contourpy.contour_generator(x=r, y=z, z=normalized, name="serial")
    axis = (float(axis_r), float(axis_z))
    if vessel is not None:
        vessel = np.asarray(vessel, float)
        if vessel.ndim != 2 or vessel.shape[1] != 2 or len(vessel) < 3 or not np.isfinite(vessel).all():
            raise ValueError("vessel must be a finite (N, 2) polygon")
        vessel = Path(np.vstack([vessel, vessel[0]]))
    c95 = _closed_contour(generator, .95, axis, vessel)
    if c95 is None:
        result["q95_error"] = "no closed 95% contour around axis" + (" inside vessel" if vessel is not None else "")
    elif f_profile.ndim != 1 or f_profile.size < 2 or not np.isfinite(f_profile).all():
        result["q95_error"] = "invalid F profile"
    else:
        gz, gr = np.gradient(psi, z, r, edge_order=2)
        gradient = np.hypot(gr, gz)
        # Subdivide straight contour edges for stable midpoint quadrature.
        fractions = (np.arange(4) + .5) / 4
        points = c95[:-1, None, :] + fractions[None, :, None] * np.diff(c95, axis=0)[:, None, :]
        points = points.reshape(-1, 2)
        lengths = np.repeat(np.linalg.norm(np.diff(c95, axis=0), axis=1) / 4, 4)
        grad = _bilinear(gradient, r, z, points)
        if np.any(~np.isfinite(grad)) or np.any(grad <= 0) or np.any(points[:, 0] <= 0):
            result["q95_error"] = "invalid gradient on 95% contour"
        else:
            f95 = np.interp(.95, np.linspace(0, 1, f_profile.size), f_profile)
            result["q95"] = float(abs(f95) / (2 * np.pi) * np.sum(lengths / (points[:, 0] * grad)))

    boundary = None
    for level in (1., 1 - 1e-6, 1 - 1e-4, .999, .995, .99):
        boundary = _closed_contour(generator, level, axis, vessel)
        if boundary is not None:
            result["boundary_level"] = level
            break
    if boundary is None:
        result["betaN_error"] = "no closed boundary around axis"
        return result
    volume, minor_radius, major_radius = _geometry(boundary)
    result["volume"] = volume
    if (volume <= 0 or minor_radius <= 0 or major_radius <= 0
            or not np.isfinite(ip) or abs(ip) < 1 or not np.isfinite(i_rod) or abs(i_rod) < 1):
        result["betaN_error"] = "invalid geometry/current"
        return result
    if p_profile.ndim != 1 or p_profile.size < 2 or not np.isfinite(p_profile).all():
        result["betaN_error"] = "invalid pressure profile"
        return result
    tolerance = max(float(np.abs(p_profile).max()), 1.) * 1e-6
    if p_profile.min() < -tolerance:
        result["betaN_error"] = "negative pressure profile"
        return result
    # Four midpoint samples per native cell; geometry normalization uses the
    # polygon's exact first area moment rather than a random native polygon.
    x = (r[:-1, None] + np.diff(r)[:, None] * np.array([.25, .75])).ravel()
    y = (z[:-1, None] + np.diff(z)[:, None] * np.array([.25, .75])).ravel()
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    inside = Path(boundary).contains_points(points)
    u = np.clip(_bilinear(normalized, r, z, points[inside]), 0, 1)
    pressure = np.interp(u, np.linspace(0, 1, p_profile.size), np.maximum(p_profile, 0))
    areas = np.outer(np.repeat(np.diff(z) / 2, 2), np.repeat(np.diff(r) / 2, 2)).ravel()
    pressure_integral = float(np.sum(pressure * 2 * np.pi * points[inside, 0] * areas[inside]))
    bt = MU0 * abs(i_rod) / (2 * np.pi * major_radius)
    beta_t = 2 * MU0 * (pressure_integral / volume) / bt ** 2
    result["betaN"] = float(100 * beta_t * minor_radius * bt / (abs(ip) / 1e6))
    return result


def fill_scalar(times, values, fallback=None):
    """按该标量自身的有效掩码插值；全炮无有效值时显式使用预测中位数。"""
    times, values = np.asarray(times, float), np.asarray(values, float)
    good = np.isfinite(values)
    if good.any():
        return np.interp(times, times[good], values[good]), int((~good).sum()), False
    if fallback is None or not np.isfinite(fallback):
        raise ValueError("entire shot lacks a valid scalar and no fallback is available")
    return np.full_like(values, float(fallback)), len(values), True
