"""Stage 2 passive-current GSFit: per-shot passive solve -> full-grid delta_all correction.

入口为 `scripts/challenge_passive.sh <步骤> --output <运行目录> [选项]`，步骤：

    solve   逐炮求解：PF 电流 + 预测的 78 路被动涡流 + MAE 磁诊断，关键帧间隔 --frame-stride，
            写 shots_gsfit/<i>.npz（已存在的炮跳过，可断点续跑）
    pack    与原版提交组合出 delta_all（见 training.challenge_variants），
            写 submission_<variant>.npz 与 pack.json

输入路径全部来自 `configs/challenge/passive_*.json`，命令行可逐项覆盖。
"""

from __future__ import annotations
import argparse
import gc
import hashlib
import json
import logging
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
WORK = REPO.parents[1]  # projects/ 与 experiments/ 所在的工作区根目录（个人目录或 public/gjy）
PATH_KEYS = [
    "source", "geometry", "passive_checkpoint", "baseline_submission", "output_grid", "envelope", "gsfit_backend",
]
NUMERIC_THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS")
_WORKER_CTX = {}


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def encode(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)

    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.json")
    tmp.write_text(json.dumps(value, indent=2, default=encode))
    tmp.replace(path)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _use_backend(backend) -> None:
    """GSFit 0.0.10 原生后端（gsfit_rs + gsfit）通过 sys.path 加载，路径来自配置。"""
    os.environ["GSFIT_GEOMETRY_CACHE"] = "1"
    for name in NUMERIC_THREADS:
        os.environ[name] = "1"
    if backend and str(backend) not in sys.path:
        sys.path.insert(0, str(backend))


# ---------------------------------------------------------------- solve
def parse_channel_weights(items) -> dict[str, float]:
    """`NAME=倍率` → {传感器名: 倍率}。倍率乘在该传感器的拟合残差行权重上（×2 即平方惩罚 ×4）。"""
    weights = {}
    for item in items or []:
        name, _, value = str(item).partition("=")
        try:
            factor = float(value)
        except ValueError:
            factor = float("nan")
        if not name or not np.isfinite(factor) or factor <= 0:
            raise ValueError(f"Bad channel weight {item!r}; expected NAME=positive_number")
        weights[name] = factor
    return weights


def weighted_fit_settings(settings: dict, name: str, included: bool, weights: dict[str, float]) -> dict:
    """只放大参与拟合的指定传感器的行权重；测量值、σ 与其他通道不变。"""
    factor = weights.get(name, 1.0) if included else 1.0
    if factor == 1.0:
        return settings
    return dict(settings, fit_settings_weight=float(settings["fit_settings_weight"]) * factor)


PROFILE_DEFAULT = dict(dof=1, reg_scale=1.0, reg_form="last", init="v4", max_iter=35)
PROFILE_INITS = {"v4": (0.9, 0.35, 1.5), "upstream": (0.8, 0.5, 2.0)}  # (R, a, κ) 初始边界


def profile_regularisation(dof: int, scale: float, form: str, base: float) -> np.ndarray:
    """p′/FF′ 正则矩阵：`last` 为原版 1×dof 只罚最高阶；`diag` 为 diag(base·scale·[1..dof])。"""
    if form == "diag":
        return np.diag(base * scale * np.arange(1.0, dof + 1))
    reg = np.zeros((1, dof))
    reg[0, -1] = base * scale
    return reg


def _init_worker(source_dir, out_dir, geom_path, ckpt_path, output_grid, backend, frame_stride,
                 channel_weights=None, passive_source="circuit", r_max=1.98, profile=None):
    _use_backend(backend)
    logging.disable(logging.CRITICAL)
    from training import challenge_gsfit as base
    from training.gsfit_eval_full import physical_channel_map

    source = Path(source_dir)
    manifest = json.loads((source / "manifest.json").read_text())
    by_shot: dict[int, list[dict]] = {}
    for c in manifest["cases"]:
        by_shot.setdefault(int(c["official_index"]), []).append(c)
    for cases in by_shot.values():
        cases.sort(key=lambda x: int(x["frame"]))

    geom = dict(np.load(geom_path, allow_pickle=False))
    with np.load(source / "case_magnetics.npz", allow_pickle=False) as mag_npz:
        channels = mag_npz["channels"].tolist()
        predictions = np.asarray(mag_npz["prediction"], dtype=np.float32)

    labels = {
        "magnetics/flux_loop_flux": [str(x) for x in geom["flux_loop_channel"]],
        "magnetics/b_field_pol_probe_ccbv_field": [str(x) for x in geom["b_field_pol_probe_ccbv_channel"]],
        "magnetics/b_field_pol_probe_obr_field": [str(x) for x in geom["b_field_pol_probe_obr_channel"]],
        "magnetics/b_field_pol_probe_obv_field": [str(x) for x in geom["b_field_pol_probe_obv_channel"]],
    }
    mapping = physical_channel_map(channels, labels)
    col_by_name = {row["name"]: row["column"] for row in mapping}

    if passive_source != "v3":
        raise ValueError("only --passive-source v3 is packaged (the circuit model is not part of this entry)")
    # v3 有监督模型逐帧预测的 pf_passive（training.challenge_v3 infer 写出）。
    with np.load(source / "case_passive.npz", allow_pickle=False) as pz:
        v3_channels = pz["channels"].tolist()
        case_passive = np.asarray(pz["prediction"], dtype=float)[
            :, [v3_channels.index(str(ch_name)) for ch_name in geom["passive_channel"]]]

    with np.load(output_grid, allow_pickle=False) as env:
        r_off, z_off = env["grid_R"].copy(), env["grid_Z"].copy()
    rr_g, zz_g = np.meshgrid(r_off, z_off)
    # 被动导体在评分网格上的平均格林函数，用于解析扣除单极磁通偏移。
    g_mean = np.array(
        [
            float(np.mean(base.green_psi_per_rad(rr_g, zz_g, float(rc), float(zc))))
            for rc, zc in zip(geom["passive_r"], geom["passive_z"])
        ],
        dtype=float,
    )
    _WORKER_CTX.update(
        source=source, out=Path(out_dir), by_shot=by_shot, geom=geom, predictions=predictions,
        col_by_name=col_by_name, r_off=r_off, z_off=z_off,
        g_mean=g_mean, frame_stride=int(frame_stride), channel_weights=dict(channel_weights or {}),
        case_passive=case_passive, r_max=float(r_max),
        profile=dict(PROFILE_DEFAULT, **(profile or {})),
    )


def solve_one_shot(shot_idx: int) -> dict:
    import gsfit_rs
    from gsfit import Gsfit
    from scipy.interpolate import RegularGridInterpolator
    from training import challenge_gsfit as base
    from training import challenge_mast as cm
    from training.gsfit_eval_full import canonical_sensor_name
    from training.gsfit_postprocess import stable_scalars

    out_dir: Path = _WORKER_CTX["out"]
    dest = out_dir / "shots_gsfit" / f"{shot_idx:04d}.npz"
    if dest.exists():
        with np.load(dest, allow_pickle=False) as d:
            conv = d["converged"]
            return dict(
                shot=shot_idx, frames=int(len(conv)), attempted=int(d["attempted_count"]),
                converged=int(np.sum(conv)), neg_p=int(d["neg_p_count"]), ts_rec=int(d["ts_rec_count"]),
                seconds=float(d["seconds"]), skipped=True,
                error=str(d["error"]) if "error" in d.files else "",
            )

    t0 = time.monotonic()
    source: Path = _WORKER_CTX["source"]
    all_shot_cases = _WORKER_CTX["by_shot"][shot_idx]
    stride = _WORKER_CTX["frame_stride"]
    if stride > 1 and len(all_shot_cases) > 2:
        sel_idx = sorted(set(range(0, len(all_shot_cases), stride)) | {len(all_shot_cases) - 1})
        shot_cases = [all_shot_cases[i] for i in sel_idx]
    else:
        shot_cases = all_shot_cases

    geom = _WORKER_CTX["geom"]
    predictions = _WORKER_CTX["predictions"]
    col_by_name = _WORKER_CTX["col_by_name"]
    r_off = _WORKER_CTX["r_off"]
    z_off = _WORKER_CTX["z_off"]
    g_mean = _WORKER_CTX["g_mean"]
    weights = _WORKER_CTX["channel_weights"]

    times = np.array([float(c["time_s"]) for c in shot_cases], dtype=float)
    case_ids = [int(c["id"]) for c in shot_cases]
    frames_idx = [int(c["frame"]) for c in shot_cases]
    n_times = int(all_shot_cases[0]["n_times"])

    with np.load(source / f"public_shots/{shot_idx:04d}.npz", allow_pickle=False) as s_npz:
        shot = {k: s_npz[k] for k in s_npz.files}
    efit_ms = np.asarray(shot["efit_ms"], dtype=float)

    turns = {}
    for c in np.asarray(shot["coil_col"]):
        s = str(c)
        turns[s] = turns.get(s, 0) + 1
    coils = gsfit_rs.Coils()
    coil_col = np.asarray(shot["coil_col"])
    for column in cm.GSFIT_PF:
        sel = coil_col == column
        if not sel.any():
            continue
        raw = cm.interp_finite(shot["mag_t"], shot[f"gs_{column}"], times)
        n = max(turns.get(column, 1), 1)
        vals = raw * 500.0 if column == "magnetics_sol_current" else raw * 1000.0 / n
        coils.add_pf_coil(
            column, shot["coil_R"][sel], shot["coil_Z"][sel],
            np.abs(shot["coil_width"][sel]), np.abs(shot["coil_height"][sel]), times, vals,
        )

    # 被动涡流：v3 模型逐帧预测。
    p_vals = _WORKER_CTX["case_passive"][case_ids]
    if not np.isfinite(p_vals).all():
        raise ValueError(f"shot {shot_idx}: v3 passive prediction has non-finite values")
    for i, name in enumerate(geom["passive_name"]):
        coils.add_pf_coil(
            str(name), geom["passive_r"][i : i + 1], geom["passive_z"][i : i + 1],
            np.abs(geom["passive_width"][i : i + 1]), np.abs(geom["passive_height"][i : i + 1]),
            times, p_vals[:, i],
        )

    tf_vals = np.abs(cm.interp_finite(shot["mag_t"], shot["tf_A"], times)) * 24.0
    tf_vals = np.where(np.isfinite(tf_vals) & (tf_vals > 0), tf_vals, 1.0e6)
    coils.add_tf_coil(times, tf_vals)

    preds = predictions[case_ids]
    bp, flux = gsfit_rs.BpProbes(), gsfit_rs.FluxLoops()
    rec_idx = 0
    for fam, angle in (("ccbv", np.pi / 2), ("obr", 0.0), ("obv", np.pi / 2)):
        prefix = f"b_field_pol_probe_{fam}"
        names = [canonical_sensor_name(x) for x in geom[f"{prefix}_channel"]]
        gnames = [str(x) for x in geom[f"{prefix}_geometry_channel"]]
        idx_g = base.geometry_indices(geom[f"{prefix}_channel"], gnames)
        for i, name in enumerate(names):
            if name not in col_by_name:
                continue
            measured = preds[:, col_by_name[name]].astype(float)
            if not np.isfinite(measured).all():
                continue  # 真实磁诊断中整炮缺测的通道；原版逐帧求解同样只用有限值（MAE 预测无缺失）
            jg = idx_g[i]
            included = base.sensor_role(rec_idx, name, 0, ["P4L/1", "P4L/4"]) == "fit"
            rec_idx += 1
            bp.add_sensor(
                name=name, geometry_r=float(geom[f"{prefix}_r"][jg]), geometry_z=float(geom[f"{prefix}_z"][jg]),
                geometry_angle_pol=angle, fit_settings_comment="MAE", fit_settings_include=included,
                time=times, measured=measured,
                **weighted_fit_settings(base.fit_settings(0.002, "bp"), name, included, weights),
            )
    fnames = [canonical_sensor_name(x) for x in geom["flux_loop_channel"]]
    fidx = base.geometry_indices(geom["flux_loop_channel"], geom["flux_loop_geometry_channel"], flux=True)
    for i, name in enumerate(fnames):
        if name not in col_by_name:
            continue
        measured = preds[:, col_by_name[name]].astype(float)
        if not np.isfinite(measured).all():
            continue
        jg = fidx[i]
        included = base.sensor_role(rec_idx, name, 0, ["P4L/1", "P4L/4"]) == "fit"
        rec_idx += 1
        flux.add_sensor(
            name=name, geometry_r=float(geom["flux_loop_r"][jg]), geometry_z=float(geom["flux_loop_z"][jg]),
            fit_settings_comment="MAE", fit_settings_include=included, time=times, measured=measured,
            **weighted_fit_settings(base.fit_settings(0.005, "flux"), name, included, weights),
        )

    wall_r, wall_z = geom["limiter_r"], geom["limiter_z"]
    ip_vals = cm.interp_finite(shot["mag_t"], shot["ip_A"], times)
    rogowski = gsfit_rs.RogowskiCoils()
    rogowski.add_sensor(
        name="plasma_current", r=wall_r, z=wall_z, fit_settings_comment="Ip", fit_settings_include=True,
        time=times, measured=ip_vals, gaps_r=np.array([]), gaps_z=np.array([]), gaps_d_r=np.array([]),
        gaps_d_z=np.array([]), gaps_name=[], **base.fit_settings(500.0, "ip"),
    )
    prof = _WORKER_CTX["profile"]
    dof = int(prof["dof"])
    reg_p = profile_regularisation(dof, float(prof["reg_scale"]), prof["reg_form"], 4.5e-7)
    reg_f = profile_regularisation(dof, float(prof["reg_scale"]), prof["reg_form"], 1.7)
    init_r, init_a, init_k = PROFILE_INITS[prof["init"]]
    plasma = gsfit_rs.Plasma(
        65, 129, 0.06, _WORKER_CTX.get("r_max", 1.98), -2.0, 2.0, np.linspace(0, 1, 65), *base.dense_limiter(wall_r, wall_z),
        wall_r, wall_z, gsfit_rs.EfitPolynomial(dof, reg_p), gsfit_rs.EfitPolynomial(dof, reg_f),
        float(np.nanmedian(ip_vals)), init_r, 0.0, init_a, init_k,
    )
    controller = Gsfit(
        pulseNo=0, run_name="CHALLENGE", settings_path="mastu_with_synthetic_data_from_freegsnke", write_to_mds=False,
    )
    controller.settings["GSFIT_code_settings.json"]["RAYON_NUM_THREADS"] = 1
    controller.set_environment_variables()
    controller.results["TIME"] = times
    controller.coils, controller.passives, controller.plasma = coils, gsfit_rs.Passives(), plasma
    controller.bp_probes, controller.flux_loops, controller.rogowski_coils = bp, flux, rogowski
    controller.isoflux, controller.isoflux_boundary = gsfit_rs.Isoflux(), gsfit_rs.IsofluxBoundary()
    controller.pressure_sensors, controller.stationary_point, controller.dialoop = (
        gsfit_rs.Pressure(), gsfit_rs.StationaryPoint(), gsfit_rs.Dialoop(),
    )
    controller.settings["GSFIT_code_settings.json"]["numerics"].update(
        n_iter_max=int(prof["max_iter"]), n_iter_min=3, n_iter_no_vertical_feedback=1, gs_error=5e-6
    )

    psi_raw_list = [None] * n_times
    psi_mono_list = [None] * n_times
    q95_arr = np.full(n_times, np.nan, dtype=np.float64)
    bn_mag_arr = np.full(n_times, np.nan, dtype=np.float64)
    bn_ts_arr = np.full(n_times, np.nan, dtype=np.float64)
    conv_mask = np.zeros(n_times, dtype=bool)
    neg_p_count = 0
    ts_rec_count = 0
    error = ""

    try:
        controller.calculate_greens()
        controller.inverse_solver_rust()
        pl = controller.plasma
        solved_times = pl.get_array1(["time"])
        if len(solved_times) > 0:
            gs_r = pl.get_array1(["grid", "r"])
            gs_z = pl.get_array1(["grid", "z"])
            psi_all = base.psi_to_weber_per_rad(pl.get_array3(["profiles_2d", "r_z", "psi"]))
            gs_err_all = pl.get_array1(["global", "gs_error"])
            ip_out_all = pl.get_array1(["global", "ip"])
            irod_all = pl.get_array1(["global", "i_rod"])
            ax_r_all = pl.get_array1(["global", "r_mag"])
            ax_z_all = pl.get_array1(["global", "z_mag"])
            psi_a_all = pl.get_array1(["global", "psi_a"]) / (2.0 * np.pi)
            psi_b_all = pl.get_array1(["boundary", "psi"]) / (2.0 * np.pi)
            f_prof_all = pl.get_array2(["profiles_1d", "psi_norm", "f"])
            p_prof_all = pl.get_array2(["profiles_1d", "psi_norm", "p"])
            vessel = np.column_stack([pl.get_array1(["vessel", "r"]), pl.get_array1(["vessel", "z"])])

            zz, rr = np.meshgrid(z_off, r_off, indexing="ij")
            pts = np.column_stack(
                [np.clip(zz, gs_z.min(), gs_z.max()).ravel(), np.clip(rr, gs_r.min(), gs_r.max()).ravel()]
            )
            mono_shifts = 0.62 * (p_vals @ g_mean)

            for k, f_idx in enumerate(frames_idx):
                match = np.flatnonzero(np.isclose(solved_times, times[k], atol=1e-6))
                if match.size == 0:
                    continue
                m = int(match[0])
                if not np.isfinite(psi_all[m]).all() or float(gs_err_all[m]) > 5e-6:
                    continue
                ts_pe = base.interp_thomson_electron_pressure(shot["ts_t"], shot["te120"], shot["ne120"], float(times[k]))
                common = (
                    psi_all[m], gs_r, gs_z, float(psi_a_all[m]), float(psi_b_all[m]), f_prof_all[m], p_prof_all[m],
                    float(ip_out_all[m]), float(irod_all[m]), float(ax_r_all[m]), float(ax_z_all[m]),
                )
                sc_mag = stable_scalars(*common, vessel=vessel)
                sc_ts = stable_scalars(*common, vessel=vessel, ts_r=cm.THOMSON_R, ts_pe=ts_pe, ts_alpha=2.0)
                psi65 = RegularGridInterpolator(
                    (gs_z, gs_r), psi_all[m], bounds_error=False, fill_value=None
                )(pts).reshape(zz.shape)
                conv_mask[f_idx] = True
                psi_raw_list[f_idx] = psi65.astype(np.float32)
                psi_mono_list[f_idx] = (psi65 - mono_shifts[k]).astype(np.float32)
                if np.isfinite(sc_mag["q95"]) and float(sc_mag["q95"]) > 0:
                    q95_arr[f_idx] = float(sc_mag["q95"])
                if np.isfinite(sc_mag["betaN"]) and float(sc_mag["betaN"]) > 0:
                    bn_mag_arr[f_idx] = float(sc_mag["betaN"])
                elif sc_mag.get("betaN_error") == "negative pressure profile":
                    neg_p_count += 1
                if np.isfinite(sc_ts["betaN"]) and float(sc_ts["betaN"]) > 0:
                    bn_ts_arr[f_idx] = float(sc_ts["betaN"])
                    if sc_ts.get("betaN_source") == "thomson_kinetic_fallback":
                        ts_rec_count += 1
    except Exception as exc:  # 单炮求解失败不影响其他炮；整炮记为未收敛，pack 时退回原版。
        error = f"{type(exc).__name__}: {exc}"
    finally:
        del controller, coils, bp, flux, rogowski, plasma
        gc.collect()

    if any(p is not None for p in psi_raw_list):
        flux_raw, _, _ = cm._fill_shot(efit_ms, psi_raw_list, q95_arr.copy(), bn_mag_arr.copy())
        flux_mono, _, _ = cm._fill_shot(efit_ms, psi_mono_list, q95_arr.copy(), bn_ts_arr.copy())
    else:
        flux_raw = np.full((n_times, 65, 65), np.nan, dtype=np.float32)
        flux_mono = np.full((n_times, 65, 65), np.nan, dtype=np.float32)

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        tmp, efit_ms=efit_ms, converged=conv_mask, attempted_count=np.array(len(shot_cases)),
        psi_raw=flux_raw, psi_mono=flux_mono, q95=q95_arr, betaN_mag=bn_mag_arr, betaN_ts=bn_ts_arr,
        neg_p_count=np.array(neg_p_count), ts_rec_count=np.array(ts_rec_count),
        seconds=np.array(time.monotonic() - t0), error=np.array(error),
    )
    tmp.replace(dest)
    return dict(
        shot=shot_idx, frames=n_times, attempted=len(shot_cases), converged=int(conv_mask.sum()),
        neg_p=neg_p_count, ts_rec=ts_rec_count, seconds=round(time.monotonic() - t0, 2), skipped=False,
        error=error,
    )


def solve(args):
    out = args.output
    (out / "shots_gsfit").mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.source / "manifest.json").read_text())
    shot_indices = sorted({int(c["official_index"]) for c in manifest["cases"]})
    if args.shots:
        shot_indices = [i for i in shot_indices if i in set(args.shots)]
    print(
        f"solve: {len(shot_indices)} shots, workers={args.workers}, stride={args.frame_stride}, "
        f"channel_weights={args.channel_weights}, passive_source={args.passive_source}, r_max={args.r_max}",
        flush=True,
    )
    started = time.monotonic()
    rows = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
        initargs=(
            str(args.source), str(out), str(args.geometry), str(args.passive_checkpoint),
            str(args.output_grid), str(args.gsfit_backend), args.frame_stride, args.channel_weights,
            args.passive_source, args.r_max, args.profile,
        ),
    ) as pool:
        futures = [pool.submit(solve_one_shot, idx) for idx in shot_indices]
        for done, fut in enumerate(as_completed(futures), 1):
            try:
                rows.append(fut.result())
            except BrokenProcessPool as exc:
                # 进程池损坏时其余任务一并失效；已写出的炮会在重跑时跳过。
                raise SystemExit(
                    f"BrokenProcessPool after {len(rows)} shots; rerun with fewer --workers (16-32). "
                    "Existing shots_gsfit/*.npz are skipped on resume."
                ) from exc
            if done % 25 == 0 or done == len(shot_indices):
                conv = sum(r["converged"] for r in rows)
                att = sum(r["attempted"] for r in rows)
                print(
                    f"solve {done}/{len(shot_indices)} | converged {conv}/{att} ({100 * conv / max(att, 1):.2f}%)"
                    f" | elapsed {time.monotonic() - started:.0f}s",
                    flush=True,
                )
    rows.sort(key=lambda r: r["shot"])
    summary = dict(
        shots=len(rows), frame_stride=args.frame_stride, channel_weights=args.channel_weights, profile=args.profile,
        passive_source=args.passive_source, r_max=args.r_max, total_frames=sum(r["frames"] for r in rows), attempted_frames=sum(r["attempted"] for r in rows),
        converged_frames=sum(r["converged"] for r in rows), neg_p_frames=sum(r["neg_p"] for r in rows),
        ts_recovered_frames=sum(r["ts_rec"] for r in rows), failed_shots=[r["shot"] for r in rows if r["error"]],
        elapsed_seconds=time.monotonic() - started, per_shot=rows,
    )
    write_json(out / "gsfit_summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_shot"}, indent=2), flush=True)


# ---------------------------------------------------------------- pack
def pack(args):
    from training import challenge_variants as V

    out = args.output
    shots_dir = out / "shots_gsfit"
    indices = sorted(int(p.stem) for p in shots_dir.glob("[0-9][0-9][0-9][0-9].npz"))
    manifest = json.loads((args.source / "manifest.json").read_text())
    expected = sorted({int(c["official_index"]) for c in manifest["cases"]})
    missing = sorted(set(expected) - set(indices))
    if missing and not args.allow_missing:
        raise RuntimeError(f"{len(missing)} shots lack shots_gsfit output (e.g. {missing[:5]}); run solve first")
    passive = {}
    for i in indices:
        with np.load(shots_dir / f"{i:04d}.npz", allow_pickle=False) as z:
            passive[i] = {k: z[k] for k in ("efit_ms", "converged", "psi_mono", "q95", "betaN_ts")}
    mask = None
    if args.envelope:  # only for --passive-variant delta_mask (reproduction of submission 1012261)
        with np.load(args.envelope, allow_pickle=False) as env:
            mask = env["mask_coarse"].astype(bool)
    names = V.VARIANTS if mask is not None else ("delta_all",)
    arrays = {name: {} for name in names}
    per_shot = []
    with np.load(args.baseline_submission, allow_pickle=False) as sub:
        for i in indices:
            k = f"shot_{i:04d}"
            base = {n: sub[f"{k}_{n}"] for n in ("psirz", "q95", "betaN")}
            if len(base["q95"]) != len(passive[i]["efit_ms"]):
                raise ValueError(f"{k}: baseline and passive frame counts differ")
            variants, used = V.build_shot(base, passive[i], mask)
            for name, value in variants.items():
                for field, arr in value.items():
                    if not np.isfinite(arr).all():
                        raise ValueError(f"{k} {name} {field}: non-finite values")
                    arrays[name][f"{k}_{field}"] = arr
            conv = passive[i]["converged"]
            per_shot.append(dict(
                index=i, frames=int(len(conv)), keyframes=int(conv.sum()), passive_used=bool(used),
                psi_delta_rms=float(np.sqrt(np.mean((variants["delta_all"]["psirz"] - base["psirz"]) ** 2))),
                q95_delta_mean=float(np.mean(variants["delta_all"]["q95"] - base["q95"])),
            ))
    files = {}
    for name in names:
        target = out / f"submission_{name}.npz"
        temp = out / f"submission_{name}.pending.npz"
        np.savez_compressed(temp, **arrays[name])
        temp.replace(target)
        files[name] = dict(path=str(target), sha256=sha256_file(target))
        print(f"pack {name}: {target}", flush=True)
    write_json(out / "pack.json", dict(
        shots=len(indices), missing_shots=missing, baseline_submission=str(args.baseline_submission),
        baseline_sha256=sha256_file(args.baseline_submission), correction=list(names),
        shots_without_passive=[r["index"] for r in per_shot if not r["passive_used"]],
        variants=files, per_shot=per_shot,
    ))


# ---------------------------------------------------------------- CLI
def configure(args):
    """读取配置并解析路径占位符 {repo}（本仓库）、{work}（工作区根目录）；命令行显式给出的路径优先。"""
    config = json.loads(args.config.read_text())
    if config.get("protocol") != "mast-v4-passive":
        raise ValueError("Expected a mast-v4-passive config")
    places = dict(repo=str(REPO), work=str(WORK))
    for key in PATH_KEYS:
        if getattr(args, key, None) is None:
            value = config["paths"].get(key)
            setattr(args, key, Path(value.format(**places)).resolve() if value else None)
    for key, value in config.get("defaults", {}).items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    # 通道权重：命令行 --channel-weight 优先，否则取配置的 "channel_weights"（默认无）。
    if getattr(args, "channel_weight", None):
        args.channel_weights = parse_channel_weights(args.channel_weight)
    else:
        args.channel_weights = {str(k): float(v) for k, v in config.get("channel_weights", {}).items()}
    args.profile = dict(PROFILE_DEFAULT, **{k: getattr(args, k) for k in PROFILE_DEFAULT
                                            if getattr(args, k, None) is not None})
    args.output = args.output.resolve()
    args.config_data = config
    args.resolved = {
        "config": str(args.config), "mode": args.mode, "output": str(args.output),
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
           if k in PATH_KEYS or k in config.get("defaults", {})},
        "channel_weights": args.channel_weights, "passive_source": args.passive_source,
        "r_max": args.r_max,
        "profile": args.profile,
    }


def _int_list(value):
    return [int(x) for x in value.split(",") if x]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["solve", "pack"])
    p.add_argument("--config", type=Path, default=REPO / "configs/challenge/passive_final.json")
    p.add_argument("--output", type=Path, required=True, help="运行目录；每次实验用新目录")
    for key in PATH_KEYS:
        p.add_argument("--" + key.replace("_", "-"), dest=key, type=Path)
    p.add_argument("--workers", type=int, help="solve 进程数，建议 16-32；96 曾导致 BrokenProcessPool")
    p.add_argument("--frame-stride", type=int)
    p.add_argument("--score-workers", type=int)
    p.add_argument("--bootstrap", type=int)
    p.add_argument("--shots", type=_int_list, help="只求解这些炮，逗号分隔（调试用）")
    p.add_argument("--allow-missing", action="store_true", help="pack 时允许部分炮缺少 passive 输出（调试用）")
    p.add_argument(
        "--channel-weight", action="append", metavar="NAME=MULT",
        help="solve 时把该传感器的残差行权重乘以 MULT，可重复，如 OBR02=2（覆盖配置的 channel_weights）",
    )
    p.add_argument("--passive-source", choices=["circuit", "v3"], default="circuit",
                   help="solve 的被动电流：circuit=线性电路模型（默认）；v3=输入目录 case_passive.npz 的 v3 预测")
    p.add_argument("--r-max", type=float, default=1.98,
                   help="GSFit 求解网格 R 上限（默认 1.98 与 v4 一致；2.0 与官方 efit_grid_R 重合）")
    p.add_argument("--dof", type=int, help="solve：p′/FF′ 自由度（默认 1）")
    p.add_argument("--reg-scale", type=float, help="solve：正则倍率（默认 1）")
    p.add_argument("--reg-form", choices=["last", "diag"], help="solve：正则矩阵形式（默认 last，只罚最高阶）")
    p.add_argument("--init", choices=sorted(PROFILE_INITS), help="solve：初始边界（默认 v4：R=0.9, a=0.35, κ=1.5）")
    p.add_argument("--max-iter", type=int, help="solve：GSFit 最大迭代数（默认 35）")
    args = p.parse_args()
    configure(args)
    for name in ("workers", "frame_stride", "score_workers", "bootstrap"):
        if getattr(args, name) is None or getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be a positive integer (config defaults or CLI)")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / f"command_{args.mode}.json", dict(argv=sys.argv, **args.resolved))
    {"solve": solve, "pack": pack}[args.mode](args)


if __name__ == "__main__":
    main()
