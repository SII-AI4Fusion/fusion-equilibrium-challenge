"""竞赛 MAST 推理：public test → MAE v2 → 磁诊断 + public test → GSFit；mapping 仅评分。

    python -m training.challenge_mast geometry
    python -m training.challenge_mast prepare --n-shots 2
    python -m training.challenge_mast infer
    gsfit-python -m training.challenge_mast gsfit
    python -m training.challenge_mast pack
    gsfit-python -m training.challenge_mast score
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

THOMSON_R = np.linspace(0.30, 1.49, 120)
GSFIT_PF = [
    "magnetics_p2l_current",
    "magnetics_p2u_current",
    "magnetics_p3l_current",
    "magnetics_p3u_current",
    "magnetics_p4l_current",
    "magnetics_p4u_current",
    "magnetics_p5l_current",
    "magnetics_p5u_current",
    "magnetics_p6l_current",
    "magnetics_p6u_current",
    "magnetics_sol_current",
]


def interp_finite(src_t, src_y, dst_t):
    src_t = np.asarray(src_t, dtype=float)
    src_y = np.asarray(src_y, dtype=float)
    dst_t = np.asarray(dst_t, dtype=float)
    ok = np.isfinite(src_t) & np.isfinite(src_y)
    if ok.sum() < 2:
        return np.full(dst_t.shape, np.nan)
    order = np.argsort(src_t[ok])
    return np.interp(dst_t, src_t[ok][order], src_y[ok][order], left=np.nan, right=np.nan)


_REPAIR_CONTEXT = {}

def _repair_inputs(output, shot_index=None):
    key = str(output)
    if key not in _REPAIR_CONTEXT:
        manifest = json.loads((output / "manifest.json").read_text())
        with np.load(output / "case_magnetics.npz", allow_pickle=False) as archive:
            magnetics = {name: archive[name] for name in archive.files}
        with np.load(output / "mast_static_geometry.npz", allow_pickle=False) as archive:
            geometry = {name: archive[name] for name in archive.files}
        assert np.array_equal(magnetics["ids"], np.arange(len(manifest["cases"])))
        _REPAIR_CONTEXT[key] = dict(manifest=manifest, magnetics=magnetics, geometry=geometry, shot_index=None)
    context = _REPAIR_CONTEXT[key]
    if shot_index is not None and context["shot_index"] != shot_index:
        with np.load(output / "public_shots" / f"{shot_index:04d}.npz", allow_pickle=False) as archive:
            context["shot"] = {name: archive[name] for name in archive.files}
        context["shot_index"] = shot_index
    return context


def _gsfit_one(payload):
    import importlib.util
    import gsfit_rs
    from scipy.interpolate import RegularGridInterpolator
    from training.gsfit_eval_full import PROTOCOL, canonical_sensor_name, physical_channel_map

    output, case_id, threads, r_off, z_off = payload
    output = Path(output)
    dest = output / "gsfit" / f"{case_id:06d}.npz"
    if dest.exists():
        with np.load(dest) as old:
            return dict(id=case_id, skipped=True, converged=bool(old["converged"]))
    context = _repair_inputs(output)
    manifest = context["manifest"]
    case = manifest["cases"][case_id]
    mag = context["magnetics"]
    j = case_id
    pred = mag["prediction"][j]
    channels = mag["channels"].tolist()
    context = _repair_inputs(output, case["official_index"])
    shot, geom = context["shot"], context["geometry"]
    t = float(case["time_s"])
    ip = float(interp_finite(shot["mag_t"], shot["ip_A"], np.array([t]))[0])
    tf = float(interp_finite(shot["mag_t"], shot["tf_A"], np.array([t]))[0])
    if not np.isfinite(ip):
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest, converged=np.array(False), error=np.array("nonfinite Ip"))
        return dict(id=case_id, converged=False, error="ip")

    labels = {
        "magnetics/flux_loop_flux": [str(x) for x in geom["flux_loop_channel"]],
        "magnetics/b_field_pol_probe_ccbv_field": [str(x) for x in geom["b_field_pol_probe_ccbv_channel"]],
        "magnetics/b_field_pol_probe_obr_field": [str(x) for x in geom["b_field_pol_probe_obr_channel"]],
        "magnetics/b_field_pol_probe_obv_field": [str(x) for x in geom["b_field_pol_probe_obv_channel"]],
    }
    mapping = physical_channel_map(channels, labels)
    provided = {
        row["name"]: float(pred[row["column"]])
        for row in mapping
        if np.isfinite(pred[row["column"]])
    }
    spec = importlib.util.spec_from_file_location("fair_mast_gsfit_base", manifest["gsfit_helper"])
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    turns = {}
    for c in np.asarray(shot["coil_col"]):
        s = str(c)
        turns[s] = turns.get(s, 0) + 1

    coils = gsfit_rs.Coils()
    coil_col = np.asarray(shot["coil_col"])
    for column in GSFIT_PF:
        sel = coil_col == column
        if not sel.any():
            continue
        raw = float(interp_finite(shot["mag_t"], shot[f"gs_{column}"], np.array([t]))[0])
        n = max(turns.get(column, 1), 1)
        value = raw * 1000.0 / n
        if column == "magnetics_sol_current":
            value = raw * 500.0  # restore SOL feed-current convention for this 656-turn geometry
        if not np.isfinite(value):
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest, converged=np.array(False), error=np.array(f"bad coil {column}"))
            return dict(id=case_id, converged=False, error=column)
        base.add_coil(
            coils,
            column,
            shot["coil_R"][sel],
            shot["coil_Z"][sel],
            shot["coil_width"][sel],
            shot["coil_height"][sel],
            value,
            t,
        )
    # parquet magnetics_tf_current is circuit kA; GSFit wants rod ampere-turns (MAST TF 24 turns).
    tf_rod = abs(tf) * 24.0 if np.isfinite(tf) and abs(tf) > 0 else 1.0e6
    coils.add_tf_coil(np.array([t]), np.array([tf_rod]))

    proto = dict(PROTOCOL)
    proto.update(time=t, passives="omit")
    args = SimpleNamespace(**proto)
    bp, flux = gsfit_rs.BpProbes(), gsfit_rs.FluxLoops()
    records = []
    for fam, angle in (("ccbv", np.pi / 2), ("obr", 0.0), ("obv", np.pi / 2)):
        prefix = f"b_field_pol_probe_{fam}"
        names = [canonical_sensor_name(x) for x in geom[f"{prefix}_channel"]]
        gnames = [str(x) for x in geom[f"{prefix}_geometry_channel"]]
        idx = base.geometry_indices(geom[f"{prefix}_channel"], gnames)
        for i, name in enumerate(names):
            if name not in provided:
                continue
            jg = idx[i]
            rec = dict(
                name=name,
                kind="bp",
                r=float(geom[f"{prefix}_r"][jg]),
                z=float(geom[f"{prefix}_z"][jg]),
                angle=angle,
                measured=provided[name],
                sigma=args.bp_sigma,
            )
            records.append(rec)
    fnames = [canonical_sensor_name(x) for x in geom["flux_loop_channel"]]
    fidx = base.geometry_indices(geom["flux_loop_channel"], geom["flux_loop_geometry_channel"], flux=True)
    for i, name in enumerate(fnames):
        if name not in provided:
            continue
        jg = fidx[i]
        records.append(
            dict(
                name=name,
                kind="flux",
                r=float(geom["flux_loop_r"][jg]),
                z=float(geom["flux_loop_z"][jg]),
                measured=provided[name],
                sigma=args.flux_sigma,
            )
        )
    for index, record in enumerate(records):
        record["calibration_gain"] = 1.0
        record["calibration_offset"] = 0.0
        record["role"] = base.sensor_role(index, record["name"], 0, PROTOCOL.get("exclude_sensor", []))
        record["included"] = record["role"] == "fit"
        common = dict(
            name=record["name"],
            geometry_r=record["r"],
            geometry_z=record["z"],
            fit_settings_comment="MAE predicted diagnostic",
            fit_settings_include=record["included"],
            time=np.array([t]),
            measured=np.array([record["measured"]]),
            **base.fit_settings(record["sigma"], record["kind"]),
        )
        if record["kind"] == "bp":
            bp.add_sensor(geometry_angle_pol=record["angle"], **common)
        else:
            flux.add_sensor(**common)
    if not any(r["kind"] == "flux" and r["included"] for r in records):
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest, converged=np.array(False), error=np.array("no flux"))
        return dict(id=case_id, converged=False, error="no flux")
    wall_r, wall_z = geom["limiter_r"], geom["limiter_z"]
    rogowski = gsfit_rs.RogowskiCoils()
    rogowski.add_sensor(
        name="plasma_current",
        r=wall_r,
        z=wall_z,
        fit_settings_comment="public test Ip",
        fit_settings_include=True,
        time=np.array([t]),
        measured=np.array([ip]),
        gaps_r=np.array([]),
        gaps_z=np.array([]),
        gaps_d_r=np.array([]),
        gaps_d_z=np.array([]),
        gaps_name=[],
        **base.fit_settings(5000.0, "ip"),
    )
    regularisation_p = np.zeros((1, args.dof))
    regularisation_f = np.zeros((1, args.dof))
    regularisation_p[0, -1] = 4.5e-7 * args.regularisation
    regularisation_f[0, -1] = 1.7 * args.regularisation
    plasma = gsfit_rs.Plasma(
        65, 129, 0.06, 1.98, -2.0, 2.0, np.linspace(0, 1, 65),
        wall_r, wall_z, wall_r, wall_z,
        gsfit_rs.EfitPolynomial(args.dof, regularisation_p),
        gsfit_rs.EfitPolynomial(args.dof, regularisation_f),
        ip, args.initial_r, 0.0, args.minor_radius, args.kappa,
    )
    controller = base.Gsfit(pulseNo=0, run_name="CHALLENGE", settings_path="mastu_with_synthetic_data_from_freegsnke", write_to_mds=False)
    controller.settings["GSFIT_code_settings.json"]["RAYON_NUM_THREADS"] = threads
    controller.set_environment_variables()
    controller.results["TIME"] = np.array([t])
    controller.coils, controller.passives, controller.plasma = coils, gsfit_rs.Passives(), plasma
    controller.bp_probes, controller.flux_loops, controller.rogowski_coils = bp, flux, rogowski
    controller.isoflux, controller.isoflux_boundary = gsfit_rs.Isoflux(), gsfit_rs.IsofluxBoundary()
    controller.pressure_sensors, controller.stationary_point, controller.dialoop = (
        gsfit_rs.Pressure(), gsfit_rs.StationaryPoint(), gsfit_rs.Dialoop(),
    )
    settings = controller.settings["GSFIT_code_settings.json"]["numerics"]
    settings.update(n_iter_max=args.max_iter, n_iter_min=3, n_iter_no_vertical_feedback=1, gs_error=5e-6)
    started = time.monotonic()
    try:
        controller.calculate_greens()
        controller.inverse_solver_rust()
        if not controller.plasma.get_vec_usize(["global", "n_iter"]):
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest, converged=np.array(False), error=np.array("no_iter"))
            return dict(id=case_id, converged=False, error="no_iter")
        result = base.extract(controller)
        if not np.isfinite(result["psi"]).all() or result["gs_error"] > 5e-6:
            raise ValueError("invalid psi")
        gs_r = np.asarray(result["r"], float)
        gs_z = np.asarray(result["z"], float)
        interp = RegularGridInterpolator(
            (gs_z, gs_r),
            np.asarray(result["psi"], float),
            bounds_error=False,
            fill_value=None,
        )
        zz, rr = np.meshgrid(z_off, r_off, indexing="ij")
        rr_c = np.clip(rr, gs_r.min(), gs_r.max())
        zz_c = np.clip(zz, gs_z.min(), gs_z.max())
        psi = interp(np.column_stack([zz_c.ravel(), rr_c.ravel()])).reshape(zz.shape)
        dest.parent.mkdir(parents=True, exist_ok=True)
        gs_psi = np.asarray(result["psi"], float)
        np.savez_compressed(
            dest,
            converged=np.array(True),
            psi=psi.astype(np.float32),
            gs_psi_native=np.asarray(result["psi"], dtype=np.float64),
            F_profile=np.asarray(result["F_profile"], dtype=np.float64),
            p_profile=np.asarray(result["p_profile"], dtype=np.float64),
            psi_axis=np.array(result["psi_axis"]), psi_boundary=np.array(result["psi_boundary"]),
            plasma_ip=np.array(result["ip"]), tf_i_rod=np.array(result["i_rod"]),
            axis_r=np.array(result["axis_r"]), axis_z=np.array(result["axis_z"]),
            native_q95=np.array(result["native_q95"]), native_betaN=np.array(result["native_betaN"]),
            postprocess_volume=np.array(result["postprocess_volume"]),
            boundary_level=np.array(result["boundary_level"]),
            q95_error=np.array(result["q95_error"]), betaN_error=np.array(result["betaN_error"]),
            q95=np.array(result.get("q_95", np.nan), dtype=np.float64),
            betaN=np.array(result.get("beta_n", np.nan), dtype=np.float64),
            gs_error=np.array(result["gs_error"], dtype=np.float64),
            seconds=time.monotonic() - started,
            gs_r=gs_r.astype(np.float32),
            gs_z=gs_z.astype(np.float32),
            gs_psi_min=np.array(np.nanmin(gs_psi), dtype=np.float64),
            gs_psi_max=np.array(np.nanmax(gs_psi), dtype=np.float64),
            interp_finite_frac=np.array(np.isfinite(psi).mean(), dtype=np.float64),
        )
        return dict(id=case_id, converged=True, seconds=time.monotonic() - started)
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest, converged=np.array(False), error=np.array(str(exc)[:300]))
        return dict(id=case_id, converged=False, error=type(exc).__name__)
    finally:
        if "controller" in locals():
            del controller


def _fill_shot(times, psi, q95, betaN):
    T = len(times)
    have = np.array([p is not None for p in psi], dtype=bool)
    if not have.any():
        return np.full((T, 65, 65), np.nan, np.float32), np.full(T, np.nan), np.full(T, np.nan)
    idx = np.flatnonzero(have)
    out = np.stack([psi[i] if psi[i] is not None else np.full((65, 65), np.nan, np.float32) for i in range(T)])
    for k in range(T):
        if have[k]:
            continue
        if k < idx[0]:
            j = idx[0]
        elif k > idx[-1]:
            j = idx[-1]
        else:
            r, l = idx[idx > k][0], idx[idx < k][-1]
            a = (times[k] - times[l]) / max(times[r] - times[l], 1e-9)
            out[k] = (1 - a) * psi[l] + a * psi[r]
            q95[k] = (1 - a) * q95[l] + a * q95[r]
            betaN[k] = (1 - a) * betaN[l] + a * betaN[r]
            continue
        out[k] = psi[j]
        q95[k] = q95[j]
        betaN[k] = betaN[j]
    return out.astype(np.float32), np.asarray(q95, np.float64), np.asarray(betaN, np.float64)
