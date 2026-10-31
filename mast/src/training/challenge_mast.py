"""MAST challenge parquet -> per-shot input NPZ and case manifest (``prepare``), plus shared helpers."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
PARQUET = None

PATCH = 0.005
WINDOW_PATCHES = 32
THOMSON_R = np.linspace(0.30, 1.49, 120)
PF_MAE = [
    ("magnetics_p2l_current", 0),
    ("magnetics_p2u_current", 1),
    ("magnetics_p2l_current", 2),
    ("magnetics_p2u_current", 3),
    ("magnetics_p3l_current", 4),
    ("magnetics_p3u_current", 5),
    ("magnetics_p4l_current", 6),
    ("magnetics_p4u_current", 7),
    ("magnetics_p5l_current", 8),
    ("magnetics_p5u_current", 9),
]
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


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))
    tmp.replace(path)


def source_identity():
    import subprocess

    names = [
        "challenge_mast.py",
        "challenge_gsfit.py",
        "gsfit_postprocess.py",
    ]
    source = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in names
    }
    status = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return dict(
        git_commit=status.stdout.strip() if status.returncode == 0 else None,
        source_sha256=source,
    )


def public_order(directory):
    """Official order is the sorted public parquet order; no reference mapping is read."""
    files = sorted(p.stem for p in directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No public-test parquet files in {directory}")
    return files


def interp_finite(src_t, src_y, dst_t):
    src_t = np.asarray(src_t, dtype=float)
    src_y = np.asarray(src_y, dtype=float)
    dst_t = np.asarray(dst_t, dtype=float)
    ok = np.isfinite(src_t) & np.isfinite(src_y)
    if ok.sum() < 2:
        return np.full(dst_t.shape, np.nan)
    order = np.argsort(src_t[ok])
    return np.interp(
        dst_t, src_t[ok][order], src_y[ok][order], left=np.nan, right=np.nan
    )


def read_parquet(stem: str) -> dict:
    import pyarrow.parquet as pq

    columns = [
        "efit_times",
        "magnetics_time",
        "magnetics_plasma_current",
        "magnetics_tf_current",
        "thomson_core_times",
        "thomson_core_R",
        "thomson_core_Te",
        "thomson_core_ne",
        "coil_input_column",
        "coil_R",
        "coil_Z",
        "coil_width",
        "coil_height",
        *GSFIT_PF,
    ]
    table = pq.read_table(PARQUET / f"{stem}.parquet", columns=columns).to_pydict()
    row = {
        k: (v[0] if isinstance(v, list) and len(v) == 1 else v)
        for k, v in table.items()
    }
    return row


def coil_turn_counts(row) -> dict[str, int]:
    cols = [str(c) for c in np.asarray(row["coil_input_column"])]
    out = {}
    for c in cols:
        out[c] = out.get(c, 0) + 1
    return out


def feed_amperes(
    row, column: str, t_s: np.ndarray, turns: dict[str, int]
) -> np.ndarray:
    mag_t = np.asarray(row["magnetics_time"], dtype=float) / 1000.0
    raw = interp_finite(mag_t, np.asarray(row[column], dtype=float), t_s)
    n = max(int(turns.get(column, 1)), 1)
    scale = 1000.0 / n if column != "magnetics_sol_current" else 1000.0
    if column == "magnetics_sol_current":
        scale = 500.0  # parquet kA ≈ 2 × TokaMark 训练用 A
    return raw * scale


def prepare(args):
    files = public_order(PARQUET)
    if (args.output / "manifest.json").exists():
        raise FileExistsError(
            "Prepared output already exists; choose a new output directory"
        )
    keys = files[: args.n_shots] if args.n_shots else files
    shots_dir = args.output / "public_shots"
    shots_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for official_index, key in enumerate(keys):
        # official_index must be global test-stream index, not sliced
        official_index = files.index(key)
        row = read_parquet(key)
        turns = coil_turn_counts(row)
        efit_ms = np.asarray(row["efit_times"], dtype=float)
        if args.frame_stride > 1:
            keep = np.arange(len(efit_ms))[:: args.frame_stride]
        else:
            keep = np.arange(len(efit_ms))
        mag_t = np.asarray(row["magnetics_time"], dtype=float) / 1000.0
        ip = (
            interp_finite(
                mag_t, np.asarray(row["magnetics_plasma_current"], float), mag_t
            )
            * 1000.0
        )
        tf = (
            interp_finite(mag_t, np.asarray(row["magnetics_tf_current"], float), mag_t)
            * 1000.0
        )
        ts_t = np.asarray(row["thomson_core_times"], dtype=float) / 1000.0
        ts_R = np.asarray(row["thomson_core_R"], dtype=float)
        te = np.asarray(row["thomson_core_Te"], dtype=float)
        ne = np.asarray(row["thomson_core_ne"], dtype=float)
        if te.ndim == 2 and te.shape[-1] == ts_R.size + 1:
            te, ne = te[:, 1:], ne[:, 1:]
        if te.ndim == 1:
            te, ne = te[None, :], ne[None, :]
        # (T_ts, nR)
        if te.shape != (ts_t.size, ts_R.size) and te.shape == (ts_R.size, ts_t.size):
            te, ne = te.T, ne.T
        te120 = np.vstack(
            [interp_finite(ts_R, te[j], THOMSON_R) for j in range(len(ts_t))]
        )
        ne120 = np.vstack(
            [interp_finite(ts_R, ne[j], THOMSON_R) for j in range(len(ts_t))]
        )
        pf = np.zeros((10, mag_t.size), dtype=np.float64)
        for col, idx in PF_MAE:
            pf[idx] = feed_amperes(row, col, mag_t, turns)
        sol = feed_amperes(row, "magnetics_sol_current", mag_t, turns)
        gs_currents = {
            c: interp_finite(mag_t, np.asarray(row[c], float), mag_t) for c in GSFIT_PF
        }
        coil_col = np.array([str(c) for c in np.asarray(row["coil_input_column"])])
        np.savez_compressed(
            shots_dir / f"{official_index:04d}.npz",
            key=np.array(key),
            efit_ms=efit_ms,
            mag_t=mag_t,
            ip_A=ip,
            tf_A=tf,
            pf_mae=pf,
            sol_mae=sol,
            ts_t=ts_t,
            te120=te120,
            ne120=ne120,
            coil_col=coil_col,
            coil_R=np.asarray(row["coil_R"], dtype=np.float64),
            coil_Z=np.asarray(row["coil_Z"], dtype=np.float64),
            coil_width=np.asarray(row["coil_width"], dtype=np.float64),
            coil_height=np.asarray(row["coil_height"], dtype=np.float64),
            **{
                f"gs_{c}": np.asarray(v, dtype=np.float64)
                for c, v in gs_currents.items()
            },
        )
        for frame in keep.tolist():
            t = float(efit_ms[frame]) / 1000.0
            t0 = t - (WINDOW_PATCHES / 2 + 0.5) * PATCH
            cases.append(
                dict(
                    id=len(cases),
                    official_index=official_index,
                    key=key,
                    frame=int(frame),
                    n_times=int(len(efit_ms)),
                    time_s=t,
                    t0=t0,
                )
            )
        print(
            f"prepare {len(keys) and files.index(key) + 1}/{len(keys)} {key} T={len(efit_ms)}",
            flush=True,
        )
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output / "manifest.json",
        dict(
            n_official=len(files),
            n_selected_shots=len(keys),
            n_cases=len(cases),
            keys=keys,
            frame_stride=args.frame_stride,
            parquet=str(PARQUET),
            geometry=str(args.output / "mast_static_geometry.npz"),
            protocol="mast-v4",
            source=source_identity(),
            cases=cases,
            mae_input="public_test D_pf_ts_ip (Ip, PF/solenoid, Thomson profiles)",
            gsfit_input="MAE magnetics_slow + public_test currents/coil geometry/Ip/TF",
        ),
    )
    print(
        json.dumps({"n_shots": len(keys), "n_cases": len(cases)}, indent=2), flush=True
    )


def _fill_shot(times, psi, q95, betaN):
    T = len(times)
    have = np.array([p is not None for p in psi], dtype=bool)
    if not have.any():
        return (
            np.full((T, 65, 65), np.nan, np.float32),
            np.full(T, np.nan),
            np.full(T, np.nan),
        )
    idx = np.flatnonzero(have)
    out = np.stack(
        [
            psi[i] if psi[i] is not None else np.full((65, 65), np.nan, np.float32)
            for i in range(T)
        ]
    )
    for k in range(T):
        if have[k]:
            continue
        if k < idx[0]:
            j = idx[0]
        elif k > idx[-1]:
            j = idx[-1]
        else:
            right, left = idx[idx > k][0], idx[idx < k][-1]
            a = (times[k] - times[left]) / max(times[right] - times[left], 1e-9)
            out[k] = (1 - a) * psi[left] + a * psi[right]
            q95[k] = (1 - a) * q95[left] + a * q95[right]
            betaN[k] = (1 - a) * betaN[left] + a * betaN[right]
            continue
        out[k] = psi[j]
        q95[k] = q95[j]
        betaN[k] = betaN[j]
    return (
        out.astype(np.float32),
        np.asarray(q95, np.float64),
        np.asarray(betaN, np.float64),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--parquet", type=Path, required=True, help="directory of mast_private_test parquet files")
    sub = p.add_subparsers(dest="mode", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("--n-shots", type=int, default=0)
    q.add_argument("--frame-stride", type=int, default=1)
    q.set_defaults(func=prepare)
    args = p.parse_args()
    if args.n_shots < 0 or args.frame_stride < 1:
        p.error("n-shots must be non-negative and frame-stride positive")
    global PARQUET
    PARQUET = args.parquet.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
