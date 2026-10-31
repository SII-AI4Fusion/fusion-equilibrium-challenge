"""传感器健康审计：从缓存生成逐炮×逐通道的健康标记和逐时间块的尖峰掩码。

FAIR-MAST v3 中部分诊断通道在某些炮段失效（无响应、增益/符号错误、整套采集失败），
数值仍然有限，``isfinite`` 掩码挡不住。本模块只读缓存，分四步：

    features  逐炮提取时间块级回归量（PF/SOL、Ip、TF 及其指数响应）、目标块均值、
              双侧尖峰计数和卡值比例 → <run>/features/<shot>.npz
    fit       在训练集放电期用 Huber-IRLS 岭回归为每个目标通道拟合“虚拟传感器”：
              输入模型只用输入回归量；磁诊断另有跨传感器模型（输入 + 锚定磁诊断，留一）
    score     逐炮把实测与虚拟传感器做线性比对：增益、相关、幅度比
    decide    按原因位给出健康标记；跨炮滚动窗口补判持续坏段；整套失效按组标记

fit/score/decide 由 ``run`` 子命令迭代两轮：第二轮拟合排除第一轮判坏的炮×通道。
磁诊断的增益/符号/一致性判据依赖等离子体位置与形状，只用线圈电流会误报，
因此这几类判定须经跨传感器模型确认，见 ``combine_magnetics``。

    PYTHONPATH=src python -m data_prep.health features --cache $CACHE --output $RUN --workers 16
    PYTHONPATH=src python -m data_prep.health run --cache $CACHE --output $RUN --workers 16

输出 ``health.npz``（原因位、评分）、``spikes.npz``（稀疏的尖峰时间块）和 ``health.json``
（阈值、计数与来源）。训练时由 ``training.dataset.CacheWindows(health=...)`` 读取，
被判坏的值按缺测处理：既不进入编码器，也不计入损失。

输入侧（PF/SOL、Ip、TF）只用卡值和尖峰两条规则，它们只依赖输入本身，比赛推理时可同样应用；
增益/相关等判据需要目标真值，只用于清洗训练与评估目标。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .build_cache import INDEX_FILE, LAYOUT_FILE
from .stats import STATS_FILE

HEALTH_FILE = "health.npz"
SPIKES_FILE = "spikes.npz"
SUMMARY_FILE = "health.json"

# 原因位。``DEFAULT_REASONS`` 为训练默认屏蔽的集合。
REASONS = {
    "dead": 1,            # 有预期响应但实测幅度 < 虚拟传感器的 10%
    "sign": 2,            # 与虚拟传感器显著负相关
    "gain": 4,            # 相关良好但增益偏离该通道总体 2 倍以上且为稳健离群
    "inconsistent": 8,    # 相关显著低于该通道总体
    "stuck": 16,          # 放电期大部分采样一阶差分为 0
    "system": 32,         # 同组过半通道 dead：整套采集失效，整组屏蔽
    "segment": 64,        # 跨炮滚动窗口内多数炮判坏，本炮证据处于边缘
}
DEFAULT_REASONS = ("dead", "sign", "gain", "inconsistent", "stuck", "system", "segment")

TARGETS = ("magnetics_slow", "pf_passive")
# 输入：(模态, 通道筛选)；只用卡值与尖峰规则。
INPUTS = ("pf_active", "summary", "tf")
SPIKE_MODALITIES = TARGETS + INPUTS

TAUS = (0.002, 0.010, 0.040)  # 指数响应时间常数 (s)，对应被动结构的 L/R 量级
MIN_IP = 50_000.0
MIN_ROWS = 8

THRESHOLDS = {
    "dead_amplitude": 0.1,
    "sign_rho": -0.3,
    "gain_factor": 2.0,
    "gain_z": 5.0,
    "gain_rho": 0.5,
    "inconsistent_rho": 0.5,
    "inconsistent_z": 5.0,
    "inconsistent_channel_rho": 0.8,
    "reliable_fraction": 0.1,
    "stuck_target": 0.5,
    "stuck_input": 0.9,
    "system_fraction": 0.5,
    "segment_window": 15,
    "segment_fraction": 0.5,
    "spike_sigma": 10.0,
    "spike_scale": 0.05,
    "spike_half_width": 8,
    "spike_max_channels": 2,
    "anchor_coverage": 0.97,
    "anchor_max_bad": 0.02,
    "spike_tolerance": 2,
}


# ---------------------------------------------------------------- 通道选择

def channel_sets(layout: dict[str, Any]) -> dict[str, list[int]]:
    """每个模态参与审计的列：输入只取电流类通道。"""
    mods = {m["name"]: m["channels"] for m in layout["modalities"]}
    sets = {
        "magnetics_slow": list(range(len(mods["magnetics_slow"]))),
        "pf_passive": list(range(len(mods["pf_passive"]))),
        "pf_active": [i for i, c in enumerate(mods["pf_active"])
                      if c.startswith("pf_active/coil_current[")],
        "summary": [mods["summary"].index("summary/ip")],
        "tf": [mods["tf"].index("tf/coil_current")],
    }
    return sets


def magnetic_groups(channels: list[str]) -> dict[str, list[int]]:
    """整套失效的判定分组；``magnetics/ip`` 不参与。"""
    groups: dict[str, list[int]] = {}
    for i, c in enumerate(channels):
        if c == "magnetics/ip":
            continue
        key = "flux_loop" if "flux_loop" in c else "pickup"
        groups.setdefault(key, []).append(i)
    return groups


# ---------------------------------------------------------------- 逐炮特征

def exponential_response(currents: np.ndarray, dt: float, taus=TAUS) -> np.ndarray:
    """``r[n] = exp(-dt/tau) r[n-1] + (I[n] - I[n-1])``，缺测增量记 0；[T, C] -> [T, C*len(taus)]。"""
    d = np.diff(currents, axis=0, prepend=currents[:1])
    d = np.where(np.isfinite(d), d, 0.0)
    out = np.empty((currents.shape[0], currents.shape[1] * len(taus)))
    for k, tau in enumerate(taus):
        a = np.exp(-dt / tau)
        r = np.zeros(currents.shape[1])
        block = out[:, k * currents.shape[1]:(k + 1) * currents.shape[1]]
        for n in range(currents.shape[0]):
            r = a * r + d[n]
            block[n] = r
    return out


def patch_mean(values: np.ndarray, spp: int, n_patches: int) -> np.ndarray:
    """[T, C] 采样 -> [P, C] 时间块均值；全缺测的块为 NaN。"""
    v = values[:n_patches * spp].reshape(n_patches, spp, -1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(v, axis=1)


def two_sided_spikes(x: np.ndarray, scale: np.ndarray, half_width: int, n_sigma: float,
                     scale_fraction: float) -> np.ndarray:
    """双侧尖峰：样本同时偏离左右两侧窗口中位数，且方向相同。

    台阶（破裂、线圈切换）只偏离一侧，不会被判为尖峰；噪声尺度用一阶差分的 MAD 估计。
    返回 [T, C] 布尔。
    """
    t, c = x.shape
    out = np.zeros((t, c), bool)
    if t < 2 * half_width + 1:
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sigma = 1.4826 * np.nanmedian(np.abs(np.diff(x, axis=0)), axis=0) / np.sqrt(2)
        threshold = n_sigma * sigma + scale_fraction * scale
        windows = np.lib.stride_tricks.sliding_window_view(x, half_width, axis=0)  # [T-h+1, C, h]
        med = np.nanmedian(windows, axis=-1)
    h = half_width
    centre = x[h:t - h]
    left = med[:t - 2 * h]          # 样本 i 之前 h 个
    right = med[h + 1:t - h + 1]    # 样本 i 之后 h 个
    dl, dr = centre - left, centre - right
    hit = ((np.abs(dl) > threshold) & (np.abs(dr) > threshold) & (np.sign(dl) == np.sign(dr)))
    out[h:t - h] = np.nan_to_num(hit, nan=False).astype(bool)
    return out


def isolated(spikes: np.ndarray, max_channels: int, tolerance: int) -> np.ndarray:
    """只保留孤立于单个/少数通道的尖峰。

    击穿、电源切换、ELM、破裂等真实瞬变会让大量通道在同一时刻同时偏离；
    传感器毛刺只出现在个别通道。同一时刻（±tolerance 采样）偏离的通道数超过
    ``max_channels`` 时视为物理事件，不算尖峰。spikes 为 [T, C] 布尔（可为多模态拼接）。
    """
    if not spikes.any():
        return spikes
    t = spikes.shape[0]
    dilated = np.zeros_like(spikes)
    for k in range(-tolerance, tolerance + 1):
        lo, hi = max(0, k), min(t, t + k)
        dilated[lo:hi] |= spikes[lo - k:hi - k]
    coherent = dilated.sum(axis=1) > max_channels
    return spikes & ~coherent[:, None]


def shot_features(cache: Path, record: dict[str, Any], sets: dict[str, list[int]],
                  scales: dict[str, np.ndarray], dt: float) -> dict[str, np.ndarray]:
    shot_dir = cache / "shots" / str(record["shot"])
    n = record["n_patches"]
    ip_patch = np.load(shot_dir / "plasma_current.npy")
    out: dict[str, np.ndarray] = {"plasma": np.abs(np.nan_to_num(ip_patch)) >= MIN_IP}
    raw: dict[str, np.ndarray] = {}
    for m, cols in sets.items():
        if m in record["modalities"]:
            arr = np.load(shot_dir / f"{m}.npy", mmap_mode="r")
            raw[m] = np.asarray(arr[:n * (arr.shape[0] // n)], np.float64)[:, cols]
        else:
            raw[m] = None
    spp = 16
    plasma_samples = np.repeat(out["plasma"], spp)
    spikes = {}
    for m in SPIKE_MODALITIES:
        cols = sets[m]
        spikes[m] = (np.zeros((n * spp, len(cols)), bool) if raw[m] is None else
                     two_sided_spikes(raw[m], scales[m][cols], THRESHOLDS["spike_half_width"],
                                      THRESHOLDS["spike_sigma"], THRESHOLDS["spike_scale"]))
    for pool in (TARGETS, INPUTS):
        joint = isolated(np.concatenate([spikes[m] for m in pool], axis=1),
                         THRESHOLDS["spike_max_channels"], THRESHOLDS["spike_tolerance"])
        start = 0
        for m in pool:
            width = spikes[m].shape[1]
            spikes[m] = joint[:, start:start + width]
            start += width
    for m in SPIKE_MODALITIES:
        cols = sets[m]
        out[f"spike_{m}"] = spikes[m].reshape(n, spp, -1).sum(axis=1).astype(np.uint8)
        if raw[m] is None:
            out[f"stuck_{m}"] = np.full(len(cols), np.nan, np.float32)
            continue
        x = raw[m]
        xp = x[plasma_samples[:len(x)]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            d = np.diff(xp, axis=0)
            fin = np.isfinite(d)
            stuck = np.where(fin.sum(0) > 0, ((d == 0) & fin).sum(0) / np.maximum(fin.sum(0), 1), np.nan)
            if m in INPUTS:
                # 输入恒为 0 是线圈未通电，属真实信息；只把非零恒值记为卡值。
                level = np.abs(np.nanmedian(xp, axis=0)) if len(xp) else np.zeros(len(cols))
                stuck = np.where(level > 0.01 * scales[m][cols], stuck, 0.0)
        out[f"stuck_{m}"] = stuck.astype(np.float32)
        if m in TARGETS:
            out[f"y_{m}"] = patch_mean(x, spp, n).astype(np.float32)
    # 回归量：13 路 PF/SOL、Ip、|TF| 的块均值，以及 PF/SOL 与 Ip 的指数响应。
    currents = []
    for m in ("pf_active", "summary"):
        currents.append(raw[m] if raw[m] is not None
                        else np.full((n * spp, len(sets[m])), np.nan))
    currents = np.concatenate(currents, axis=1)
    tf = raw["tf"] if raw["tf"] is not None else np.full((n * spp, 1), np.nan)
    response = exponential_response(currents, dt)
    regressors = np.concatenate([currents, np.abs(tf), response], axis=1)
    out["x"] = patch_mean(regressors, spp, n).astype(np.float32)
    return out


def _features_job(task: tuple[str, dict[str, Any], dict[str, list[int]], dict[str, list[float]], float, str]) -> int:
    cache, record, sets, scales, dt, output = task
    path = Path(output) / "features" / f"{record['shot']}.npz"
    if path.is_file():
        return 0
    feats = shot_features(Path(cache), record, sets,
                          {k: np.asarray(v) for k, v in scales.items()}, dt)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **feats)
    tmp.replace(path)
    return 1


# ---------------------------------------------------------------- 虚拟传感器

def huber_ridge(x: np.ndarray, y: np.ndarray, alpha: float = 1e-3, iterations: int = 4,
                c: float = 2.0) -> tuple[np.ndarray, float]:
    """标准化后的 Huber-IRLS 岭回归；返回 (系数, 截距)。x 需已标准化。"""
    xb = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    w = np.ones(len(x))
    reg = alpha * len(x) * np.eye(xb.shape[1])
    reg[-1, -1] = 0.0
    coef = np.zeros(xb.shape[1])
    for _ in range(iterations):
        xw = xb * w[:, None]
        coef = np.linalg.solve(xw.T @ xb + reg, xw.T @ y)
        r = y - xb @ coef
        s = 1.4826 * np.median(np.abs(r)) + 1e-12
        w = np.minimum(1.0, c * s / np.maximum(np.abs(r), 1e-12))
    return coef[:-1], float(coef[-1])


def load_features(output: Path, shots: list[int]) -> dict[int, dict[str, np.ndarray]]:
    data = {}
    for s in shots:
        with np.load(output / "features" / f"{s}.npz") as z:
            data[s] = {k: z[k] for k in z.files}
    return data


def design(f: dict[str, np.ndarray], anchors: list[int] | None,
           anchor_bad: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """回归行（放电期且回归量全部有限）与回归量。

    ``anchors`` 为 None 时只用输入回归量；否则追加这些磁诊断通道的块均值作为
    跨传感器回归量（留一由拟合时把目标自身的列置零实现）。``anchor_bad`` 为本炮已判坏的
    锚定通道，置为缺测，使对应行不参与。
    """
    x = f["x"]
    if anchors is not None:
        y = f.get("y_magnetics_slow")
        if y is None:
            return np.zeros(len(x), bool), x
        a = y[:, anchors].astype(np.float32).copy()
        a[f["spike_magnetics_slow"][:, anchors] > 0] = np.nan
        if anchor_bad is not None and anchor_bad.any():
            a[:, anchor_bad] = np.nan
        x = np.concatenate([x, a], axis=1)
    return f["plasma"] & np.isfinite(x).all(axis=1), x


def fit_sensors(feats: dict[int, dict[str, np.ndarray]], train_shots: list[int],
                bad: dict[str, dict[int, np.ndarray]] | None, targets=TARGETS,
                anchors: list[int] | None = None, max_rows: int = 400_000,
                seed: int = 0) -> dict[str, Any]:
    """每个目标通道一个线性虚拟传感器；``bad[m][shot]`` 为上一轮判坏的通道布尔。

    ``anchors`` 给出时为跨传感器模型（只用于 magnetics_slow），目标是锚定通道之一时
    该列不参与拟合（留一）。
    """
    rows_x, rows_y = [], {m: [] for m in targets}
    n_inputs = None
    for s in train_shots:
        f = feats[s]
        anchor_bad = bad["magnetics_slow"][s][anchors] if (anchors is not None and bad) else None
        keep, x = design(f, anchors, anchor_bad)
        n_inputs = f["x"].shape[1]
        if keep.sum() == 0:
            continue
        rows_x.append(x[keep])
        for m in targets:
            y = f.get(f"y_{m}")
            if y is None:
                y = np.full((keep.sum(), f[f"stuck_{m}"].shape[0]), np.nan)
            else:
                y = y[keep].copy()
                y[f[f"spike_{m}"][keep] > 0] = np.nan
                if bad is not None:
                    y[:, bad[m][s]] = np.nan
            rows_y[m].append(y)
    x = np.concatenate(rows_x).astype(np.float64)
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(x), size=min(max_rows, len(x)), replace=False))
    x = x[pick]
    mean, std = x.mean(0), x.std(0)
    std[std == 0] = 1.0
    xs = (x - mean) / std
    model: dict[str, Any] = {"mean": mean, "std": std,
                             "anchors": np.array(anchors if anchors is not None else [], int)}
    for m in targets:
        y = np.concatenate(rows_y[m]).astype(np.float64)[pick]
        coef = np.full((y.shape[1], x.shape[1]), np.nan)
        icpt = np.full(y.shape[1], np.nan)
        for j in range(y.shape[1]):
            ok = np.isfinite(y[:, j])
            if ok.sum() < 1000 or np.std(y[ok, j]) == 0:
                continue
            use = np.ones(x.shape[1], bool)
            if anchors is not None and j in anchors:
                use[n_inputs + anchors.index(j)] = False
            coef[j] = 0.0
            coef[j, use], icpt[j] = huber_ridge(xs[ok][:, use], y[ok, j])
        model[f"coef_{m}"] = coef
        model[f"intercept_{m}"] = icpt
    return model


def score_shot(f: dict[str, np.ndarray], model: dict[str, Any], m: str,
               anchor_bad: np.ndarray | None = None) -> np.ndarray:
    """返回 [C, 6]：(行数, 增益, 相关, 幅度比, 预测标准差, 截距)。"""
    coef, icpt = model[f"coef_{m}"], model[f"intercept_{m}"]
    out = np.full((coef.shape[0], 6), np.nan)
    if f"y_{m}" not in f:
        return out
    anchors = list(model["anchors"]) if len(model["anchors"]) else None
    keep, x = design(f, anchors, anchor_bad)
    if keep.sum() < MIN_ROWS:
        return out
    xs = (x[keep] - model["mean"]) / model["std"]
    pred = xs @ np.nan_to_num(coef).T + np.nan_to_num(icpt)
    y = f[f"y_{m}"][keep].astype(np.float64)
    y = np.where(f[f"spike_{m}"][keep] > 0, np.nan, y)
    for j in range(coef.shape[0]):
        ok = np.isfinite(y[:, j])
        if not np.isfinite(icpt[j]) or ok.sum() < MIN_ROWS:
            continue
        yy, pp = y[ok, j], pred[ok, j]
        sp, sy = pp.std(), yy.std()
        if sp == 0:
            continue
        cov = np.mean((yy - yy.mean()) * (pp - pp.mean()))
        rho = cov / (sp * sy) if sy > 0 else 0.0
        gain = cov / sp ** 2
        out[j] = (ok.sum(), gain, rho, sy / sp, sp, yy.mean() - gain * pp.mean())
    return out


def choose_anchors(feats: dict[int, dict[str, np.ndarray]], shots: list[int], channels: list[str],
                   bad: dict[int, np.ndarray], coverage: float, max_bad: float) -> list[int]:
    """跨传感器回归的锚定通道：几乎每炮都有观测且很少被判坏的磁诊断。"""
    n = len(shots)
    observed = np.zeros(len(channels))
    flagged = np.zeros(len(channels))
    for s in shots:
        f = feats[s]
        y = f.get("y_magnetics_slow")
        if y is None:
            continue
        observed += (np.isfinite(y[f["plasma"]]).sum(axis=0) >= MIN_ROWS)
        flagged += bad[s]
    return [j for j, c in enumerate(channels) if c != "magnetics/ip"
            and observed[j] >= coverage * n and flagged[j] <= max_bad * n]


# ---------------------------------------------------------------- 判定

def robust_z(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref = reference[np.isfinite(reference)]
    if len(ref) < 20:
        return np.zeros_like(values)
    med = np.median(ref)
    mad = 1.4826 * np.median(np.abs(ref - med)) + 1e-9
    return (values - med) / mad


def decide_target(scores: np.ndarray, stuck: np.ndarray, train: np.ndarray,
                  thresholds: dict[str, float] = THRESHOLDS) -> np.ndarray:
    """scores [S, C, 6] 按 ``score_shot`` 排列（炮按炮号升序），stuck [S, C]，返回原因位 [S, C]。"""
    t = thresholds
    n_shots, n_ch = scores.shape[:2]
    reason = np.zeros((n_shots, n_ch), np.uint16)
    gain, rho, amp, sp = scores[..., 1], scores[..., 2], scores[..., 3], scores[..., 4]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for j in range(n_ch):
            typical = np.nanmedian(sp[train, j]) if np.isfinite(sp[train, j]).any() else np.nan
            if not np.isfinite(typical):
                continue
            reliable = np.isfinite(gain[:, j]) & (sp[:, j] >= t["reliable_fraction"] * typical)
            dead = reliable & (amp[:, j] < t["dead_amplitude"])
            sign = reliable & ~dead & (rho[:, j] < t["sign_rho"]) & (gain[:, j] < 0)
            ref = reliable & train & ~dead & (gain[:, j] > 0)
            log_gain = np.log(np.where(gain[:, j] > 0, gain[:, j], np.nan))
            zg = robust_z(log_gain, log_gain[ref])
            centre = np.nanmedian(log_gain[ref]) if ref.any() else 0.0
            off = np.abs(log_gain - centre) > np.log(t["gain_factor"])
            gain_bad = (reliable & ~dead & (gain[:, j] > 0) & (rho[:, j] > t["gain_rho"])
                        & (np.abs(zg) > t["gain_z"]) & off)
            fisher = np.arctanh(np.clip(rho[:, j], -0.999, 0.999))
            zr = robust_z(fisher, fisher[ref])
            channel_rho = np.nanmedian(rho[ref, j]) if ref.any() else np.nan
            inconsistent = (reliable & ~dead & ~sign & (rho[:, j] < t["inconsistent_rho"])
                            & (zr < -t["inconsistent_z"]) & (channel_rho > t["inconsistent_channel_rho"]))
            reason[dead, j] |= REASONS["dead"]
            reason[sign, j] |= REASONS["sign"]
            reason[gain_bad, j] |= REASONS["gain"]
            reason[inconsistent, j] |= REASONS["inconsistent"]
    stuck_bad = np.nan_to_num(stuck, nan=0.0) > t["stuck_target"]
    reason[stuck_bad] |= REASONS["stuck"]
    return reason


def system_failure(reason: np.ndarray, scored: np.ndarray, groups: dict[str, list[int]],
                   fraction: float) -> np.ndarray:
    """同组过半已评分通道 dead/stuck 时，整组记 system。"""
    out = reason.copy()
    dead = (reason & (REASONS["dead"] | REASONS["stuck"])) > 0
    for cols in groups.values():
        n_scored = scored[:, cols].sum(axis=1)
        frac = dead[:, cols].sum(axis=1) / np.maximum(n_scored, 1)
        hit = (n_scored >= 3) & (frac >= fraction)
        out[np.ix_(hit, cols)] |= REASONS["system"]
    return out


def segment_flags(reason: np.ndarray, scores: np.ndarray, window: int, fraction: float) -> np.ndarray:
    """炮号升序下的滚动窗口：窗口内已评分炮多数判坏，且本炮证据处于边缘，补记 segment。

    边缘证据：幅度比 < 0.5、相关 < 0.9 或增益偏离 1.5 倍以上（相对该通道中位数）。
    窗口只统计有评分的炮，缺测炮不稀释比例。
    """
    out = reason.copy()
    base = REASONS["dead"] | REASONS["sign"] | REASONS["gain"] | REASONS["inconsistent"]
    bad = (reason & base) > 0
    gain, rho, amp = scores[..., 1], scores[..., 2], scores[..., 3]
    half = window // 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for j in range(reason.shape[1]):
            scored = np.where(np.isfinite(gain[:, j]))[0]
            if len(scored) < window:
                continue
            b = bad[scored, j].astype(float)
            csum = np.concatenate([[0.0], np.cumsum(b)])
            lo = np.clip(np.arange(len(scored)) - half, 0, len(scored))
            hi = np.clip(np.arange(len(scored)) + half + 1, 0, len(scored))
            frac = (csum[hi] - csum[lo]) / (hi - lo)
            med_gain = np.nanmedian(gain[scored, j][gain[scored, j] > 0])
            g = gain[scored, j] / med_gain
            borderline = ((amp[scored, j] < 0.5) | (rho[scored, j] < 0.9)
                          | ~(np.abs(np.log(np.where(g > 0, g, 1e-6))) < np.log(1.5)))
            hit = scored[(frac >= fraction) & borderline & ~bad[scored, j]]
            out[hit, j] |= REASONS["segment"]
    return out


# ---------------------------------------------------------------- 流程

def _load_context(cache: Path) -> tuple[dict, dict, dict, list[dict]]:
    layout = json.loads((cache / LAYOUT_FILE).read_text())
    stats = json.loads((cache / STATS_FILE).read_text())["modalities"]
    index = json.loads((cache / INDEX_FILE).read_text())
    sets = channel_sets(layout)
    return layout, stats, sets, sorted(index["shots"], key=lambda r: int(r["shot"]))


def run_features(args: argparse.Namespace) -> None:
    cache, output = Path(args.cache), Path(args.output)
    layout, stats, sets, records = _load_context(cache)
    (output / "features").mkdir(parents=True, exist_ok=True)
    scales = {m: stats[m]["scale"] for m in sets}
    dt = layout["patch_seconds"] / 16
    if args.shots:
        wanted = set(args.shots)
        records = [r for r in records if int(r["shot"]) in wanted]
    tasks = [(str(cache), r, sets, scales, dt, str(output)) for r in records]
    start, done = time.time(), 0
    with ProcessPoolExecutor(args.workers) as pool:
        for i, n in enumerate(pool.map(_features_job, tasks, chunksize=8)):
            done += n
            if (i + 1) % 1000 == 0:
                print(f"features {i + 1}/{len(tasks)} ({time.time() - start:.0f}s)", flush=True)
    print(f"features done: {done} new, {len(tasks)} total", flush=True)


def _score_job(task: tuple[str, int, dict[str, Any], list[str], np.ndarray | None]) -> dict[str, np.ndarray]:
    output, shot, model, targets, anchor_bad = task
    with np.load(Path(output) / "features" / f"{shot}.npz") as z:
        f = {k: z[k] for k in z.files}
    return {m: score_shot(f, model, m, anchor_bad) for m in targets}


def score_all(output: Path, shots: list[int], model: dict[str, Any], targets, workers: int,
              anchor_bad: dict[int, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    tasks = [(str(output), s, model, list(targets), None if anchor_bad is None else anchor_bad[s])
             for s in shots]
    with ProcessPoolExecutor(workers) as pool:
        scored = list(pool.map(_score_job, tasks, chunksize=32))
    return {m: np.stack([x[m] for x in scored]) for m in targets}


def decide_all(m: str, scores: np.ndarray, stuck: np.ndarray, train: np.ndarray,
               channels: list[str]) -> np.ndarray:
    rs = decide_target(scores, stuck, train)
    if m == "magnetics_slow":
        rs[:, channels.index("magnetics/ip")] = 0
        groups = magnetic_groups(channels)
    else:
        groups = {"passive": list(range(len(channels)))}
    rs = system_failure(rs, np.isfinite(scores[..., 1]), groups, THRESHOLDS["system_fraction"])
    return segment_flags(rs, scores, int(THRESHOLDS["segment_window"]), THRESHOLDS["segment_fraction"])


def combine_magnetics(input_only: np.ndarray, cross: np.ndarray, cross_scored: np.ndarray) -> np.ndarray:
    """磁诊断的最终原因位。

    gain/sign/inconsistent/segment 只在跨传感器模型（邻近探针包含等离子体形状信息）确认时保留；
    跨传感器无法评分的炮×通道只保留输入模型的 dead/stuck/system。dead/stuck/system 两个模型取并集。
    """
    robust = REASONS["dead"] | REASONS["stuck"] | REASONS["system"]
    shape = REASONS["sign"] | REASONS["gain"] | REASONS["inconsistent"] | REASONS["segment"]
    out = (input_only | cross) & robust
    out |= np.where(cross_scored, cross & shape, 0).astype(np.uint16)
    return out.astype(np.uint16)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parent, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_audit(args: argparse.Namespace) -> None:
    cache, output = Path(args.cache), Path(args.output)
    layout, stats, sets, records = _load_context(cache)
    shots = [int(r["shot"]) for r in records]
    split = np.array([r["split"] for r in records])
    train = split == "train"
    mods = {m["name"]: m["channels"] for m in layout["modalities"]}
    print("loading features", flush=True)
    feats = load_features(output, shots)

    # 输入与目标的卡值、尖峰。
    stuck = {m: np.stack([feats[s][f"stuck_{m}"] for s in shots]) for m in SPIKE_MODALITIES}
    applied = sum(REASONS[k] for k in DEFAULT_REASONS)
    train_shots = [s for s, tr in zip(shots, train) if tr]
    rounds = []

    def as_bad(reasons: dict[str, np.ndarray]) -> dict[str, dict[int, np.ndarray]]:
        return {m: {s: (reasons[m][i] & applied) > 0 for i, s in enumerate(shots)} for m in reasons}

    # 1) 输入模型：线圈电流、Ip、TF 及其指数响应 → 每个目标通道。
    bad_prev = None
    for r in range(args.rounds):
        t0 = time.time()
        model = fit_sensors(feats, train_shots, bad_prev, seed=r)
        scores = score_all(output, shots, model, TARGETS, args.workers)
        reasons = {m: decide_all(m, scores[m], stuck[m], train, mods[m]) for m in TARGETS}
        bad_prev = as_bad(reasons)
        counts = {m: int(((reasons[m] & applied) > 0).sum()) for m in TARGETS}
        rounds.append({"stage": "inputs", "round": r, "bad_shot_channels": counts,
                       "seconds": round(time.time() - t0, 1)})
        print(f"inputs round {r}: {counts} ({time.time() - t0:.0f}s)", flush=True)
    input_scores, input_reasons = scores, reasons

    # 2) 跨传感器模型（磁诊断）：输入 + 锚定磁诊断通道，留一预测。
    m = "magnetics_slow"
    anchors = choose_anchors(feats, shots, mods[m], bad_prev[m], THRESHOLDS["anchor_coverage"],
                             THRESHOLDS["anchor_max_bad"])
    print(f"anchors: {len(anchors)} channels", flush=True)
    bad_cross = {m: bad_prev[m]}
    for r in range(args.rounds):
        t0 = time.time()
        cross_model = fit_sensors(feats, train_shots, bad_cross, targets=(m,), anchors=anchors, seed=r)
        anchor_bad = {s: bad_cross[m][s][anchors] for s in shots}
        cross_scores = score_all(output, shots, cross_model, (m,), args.workers, anchor_bad)[m]
        cross_reasons = decide_all(m, cross_scores, stuck[m], train, mods[m])
        combined = combine_magnetics(input_reasons[m], cross_reasons, np.isfinite(cross_scores[..., 1]))
        bad_cross = {m: {s: (combined[i] & applied) > 0 for i, s in enumerate(shots)}}
        rounds.append({"stage": "cross", "round": r,
                       "bad_shot_channels": int(((combined & applied) > 0).sum()),
                       "seconds": round(time.time() - t0, 1)})
        print(f"cross round {r}: {int(((combined & applied) > 0).sum())} ({time.time() - t0:.0f}s)", flush=True)
    reasons = {m: combined, "pf_passive": input_reasons["pf_passive"]}
    scores = {"magnetics_slow": input_scores["magnetics_slow"], "pf_passive": input_scores["pf_passive"]}

    # 输入的卡值位。
    for m in INPUTS:
        rs = np.zeros((len(shots), len(mods[m])), np.uint16)
        hit = np.nan_to_num(stuck[m], nan=0.0) > THRESHOLDS["stuck_input"]
        sub = np.zeros_like(rs, bool)
        sub[:, sets[m]] = hit
        rs[sub] |= REASONS["stuck"]
        reasons[m] = rs

    arrays: dict[str, np.ndarray] = {"shots": np.array(shots), "split": split}
    for m, rs in reasons.items():
        arrays[f"reason_{m}"] = rs
        arrays[f"channels_{m}"] = np.array(mods[m])
    for m in TARGETS:
        arrays[f"scores_{m}"] = scores[m].astype(np.float32)
        arrays[f"stuck_{m}"] = stuck[m]
    arrays["scores_cross_magnetics_slow"] = cross_scores.astype(np.float32)
    arrays["reason_inputs_magnetics_slow"] = input_reasons["magnetics_slow"]
    arrays["reason_cross_magnetics_slow"] = cross_reasons
    arrays["anchors_magnetics_slow"] = np.array(anchors)
    np.savez_compressed(output / HEALTH_FILE, **arrays)

    # 尖峰：稀疏 (炮序号, 时间块, 模态内通道)。
    spikes: dict[str, np.ndarray] = {}
    for m in SPIKE_MODALITIES:
        rows = []
        for i, s in enumerate(shots):
            p, c = np.nonzero(feats[s][f"spike_{m}"])
            if len(p):
                rows.append(np.stack([np.full(len(p), i), p, np.array(sets[m])[c]], axis=1))
        spikes[m] = np.concatenate(rows) if rows else np.zeros((0, 3), np.int64)
    np.savez_compressed(output / SPIKES_FILE, shots=np.array(shots), **spikes)
    np.savez(output / "sensor_model.npz", **{k: np.asarray(v) for k, v in model.items()})
    np.savez(output / "sensor_model_cross.npz", **{k: np.asarray(v) for k, v in cross_model.items()})

    counts = {}
    for m, rs in reasons.items():
        counts[m] = {k: int(((rs & v) > 0).sum()) for k, v in REASONS.items()}
        counts[m]["spike_patches"] = int(len(spikes.get(m, [])))
    summary = {"cache": str(cache), "git_commit": git_commit(), "shots": len(shots),
               "reasons": REASONS, "default_reasons": list(DEFAULT_REASONS),
               "thresholds": THRESHOLDS, "taus_seconds": list(TAUS), "min_ip_A": MIN_IP,
               "rounds": rounds, "counts": counts,
               "anchors": [mods["magnetics_slow"][j] for j in anchors]}
    (output / SUMMARY_FILE).write_text(json.dumps(summary, indent=1))
    print(json.dumps(counts, indent=1), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("features", "run"):
        p = sub.add_parser(name)
        p.add_argument("--cache", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
        if name == "features":
            p.add_argument("--shots", type=int, nargs="*")
        else:
            p.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    (run_features if args.command == "features" else run_audit)(args)


if __name__ == "__main__":
    main()
