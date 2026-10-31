"""冻结 v4 GSFit（无被动结构）逐帧重建，可选放大指定诊断通道的拟合行权重。

诊断实验用：在 MAE 预测或真实磁诊断两种输入上复现 v4 原版求解，并检验通道加权（如 OBR02×2）。
冻结 v4 源码有自己的 `training` 包，与本仓库的同名包冲突，所以本脚本独立运行，sys.path 只指向冻结目录。

    python scripts/challenge_v4_weighted.py stage --input <输入目录> --output <运行目录> [--channel-weight OBR02=2]
    python scripts/challenge_v4_weighted.py solve --output <运行目录> [--rank r --shards n] --workers 16 [--ids ids.json]
    python scripts/challenge_v4_weighted.py pack  --output <运行目录>

输入目录需含 manifest.json、case_magnetics.npz、mast_static_geometry.npz、public_shots/（v4 `prepare` 产物，或
真实磁诊断实验的 `real/`）。stage 用符号链接引用输入，写 config.json；solve 把逐帧结果写到 gsfit/<id>.npz
（已存在的跳过，可续跑，可多节点分片）；pack 按 v4 规则补缺（同炮 ψ 插值、标量独立插值、全体中位数兜底），
写 submission_mast.npz 与 pack.json。评分用 `scripts/challenge_passive.sh score --variants baseline`。

通道加权与 lwy gsfit_channel_weights 相同：只把参与拟合的指定传感器的 fit_settings_weight 乘以倍率
（×2 即平方惩罚 ×4），测量值、σ、掩码与其余设置不变。

GSFit 剖面设置（stage 时写入 config.json 的 protocol，缺省与冻结 v4 完全相同）：

    --dof 2 --reg-form diag --reg-scale 100 --init upstream

`--reg-form last` 为 v4 原版（正则矩阵 1×dof，只罚最高阶系数 4.5e-7·倍率 / 1.7·倍率）；`diag` 为对角
Tikhonov diag(s·[1..dof])，s 同上，所有阶都受约束。`--init upstream` 把初值 (R=0.9, a=0.35, κ=1.5) 换成
GSFit 上游默认 (R=0.8, a=0.5, κ=2.0)。pack 可加 `--fallback <另一运行目录>`：本运行未收敛的帧改用该运行
（通常是 dof=1 原版）已收敛的逐帧结果，再走 v4 补缺。
"""

from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import types

V4_DEFAULT = Path(__file__).resolve().parents[1] / "frozen_v4"
INPUTS = ("manifest.json", "case_magnetics.npz", "mast_static_geometry.npz", "public_shots")
HOOK_OLD = '            **base.fit_settings(record["sigma"], record["kind"]),\n'
HOOK_NEW = '            **_channel_weighted_fit_settings(base, record),\n'
GRID_OLD = "        65, 129, 0.06, 1.98, -2.0, 2.0, np.linspace(0, 1, 65),\n"
HOOK_FN = '''

CHANNEL_WEIGHTS = {}


def _channel_weighted_fit_settings(base, record):
    settings = base.fit_settings(record["sigma"], record["kind"])
    factor = CHANNEL_WEIGHTS.get(record["name"], 1.0) if record["included"] else 1.0
    if factor == 1.0:
        return settings
    return dict(settings, fit_settings_weight=float(settings["fit_settings_weight"]) * factor)
'''

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ["GSFIT_GEOMETRY_CACHE"] = "1"
import numpy as np  # noqa: E402

C = None


def write_json(path: Path, value) -> None:
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.json")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def parse_weights(items) -> dict[str, float]:
    weights = {}
    for item in items or []:
        name, _, value = item.partition("=")
        factor = float(value)
        if not name or not np.isfinite(factor) or factor <= 0:
            raise SystemExit(f"Bad --channel-weight {item!r}; expected NAME=positive_number")
        weights[name] = factor
    return weights


def load_frozen(v4: Path, weights: dict[str, float], r_max: float = 1.98):
    """导入冻结 v4 的 training.challenge_mast，仅在拟合权重一行挂钩（weights 为空时与原版逐字节等价的计算）。"""
    global C
    sys.path[:0] = [str(v4 / "src"), str(v4)]
    import training

    source_path = v4 / "src/training/challenge_mast.py"
    text = source_path.read_text()
    if text.count(HOOK_OLD) != 1:
        raise SystemExit("Frozen v4 fit_settings line not found exactly once; source changed?")
    module = types.ModuleType("training.challenge_mast")
    module.__file__ = str(source_path)
    module.__package__ = "training"
    sys.modules["training.challenge_mast"] = module
    training.challenge_mast = module
    text = text.replace(HOOK_OLD, HOOK_NEW)
    if float(r_max) != 1.98:
        # 求解网格 R 上限（与 gsfit-rgrid 分支 challenge_v4_rgrid.py 相同）：2.0 时 R 节点与官方 efit_grid_R 重合。
        if text.count(GRID_OLD) != 1:
            raise SystemExit("Frozen v4 Plasma grid line not found exactly once; source changed?")
        text = text.replace(GRID_OLD, f"        65, 129, 0.06, {float(r_max)!r}, -2.0, 2.0, np.linspace(0, 1, 65),\n")
    exec(compile(text + HOOK_FN, str(source_path), "exec"), module.__dict__)
    module.CHANNEL_WEIGHTS.update(weights)
    C = module
    return module


_SAVEZ = np.savez_compressed


def atomic_save(path, *args, **kwargs):
    path = Path(path)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    _SAVEZ(tmp, *args, **kwargs)
    tmp.replace(path)


def _init_worker():
    from runtime_config import configure

    configure()
    C.np.savez_compressed = atomic_save


def stage(args):
    out = args.output
    if (out / "config.json").exists():
        raise SystemExit(f"{out} already staged; use a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    for name in INPUTS:
        src = (args.input / name).resolve()
        if not src.exists():
            raise SystemExit(f"Missing input {src}")
        (out / name).symlink_to(src)
    manifest = json.loads((out / "manifest.json").read_text())
    if "gsfit_helper" not in manifest:
        # Final 输入的 manifest 没有 gsfit_helper；冻结 v4 _gsfit_one 需要它，写一份带 v4 helper 的副本。
        manifest["gsfit_helper"] = str((args.v4 / "stable_helper.py").resolve())
        (out / "manifest.json").unlink()
        write_json(out / "manifest.json", manifest)
    with np.load(out / "case_magnetics.npz", allow_pickle=False) as z:
        assert np.array_equal(z["ids"], np.arange(len(manifest["cases"])))
    write_json(out / "config.json", dict(
        input=str(args.input.resolve()), v4=str(args.v4.resolve()), channel_weights=parse_weights(args.channel_weight),
        cases=len(manifest["cases"]), gsfit_input=manifest.get("gsfit_input"),
        hook="frozen v4 _gsfit_one; fit_settings_weight × channel weight for included sensors only",
        **({"protocol": protocol_from_args(args)} if args.dof is not None else {}),
        **({"r_max": args.r_max} if args.r_max != 1.98 else {}),
    ))
    print(f"staged {len(manifest['cases'])} cases -> {out}", flush=True)


INITS = {"v4": {}, "upstream": dict(initial_r=0.8, minor_radius=0.5, kappa=2.0)}


def protocol_from_args(args) -> dict:
    return dict(dof=args.dof, regularisation=args.reg_scale, reg_form=args.reg_form, init=args.init,
                max_iter=args.max_iter)


def apply_protocol(protocol: dict | None) -> dict:
    """把 config.json 的 protocol 写进冻结 v4 的 PROTOCOL；diag 时替换 EfitPolynomial 的正则矩阵。

    须在 load_frozen 之后、fork 工作进程之前调用（子进程继承修改）。protocol 为空时什么都不改。"""
    if not protocol:
        return {}
    import gsfit_rs
    from training.gsfit_eval_full import PROTOCOL

    PROTOCOL.update(dof=int(protocol["dof"]), regularisation=float(protocol["regularisation"]),
                    max_iter=int(protocol["max_iter"]), **INITS[protocol["init"]])
    if protocol["reg_form"] == "diag" and not getattr(gsfit_rs, "_diag_patched", False):
        real = gsfit_rs.EfitPolynomial

        def diag_poly(n_dof, regs):
            scale = float(np.asarray(regs)[0, -1])
            return real(n_dof, np.diag(scale * np.arange(1.0, n_dof + 1)))

        gsfit_rs.EfitPolynomial = diag_poly
        gsfit_rs._diag_patched = True
    return dict(PROTOCOL)


def solve(args):
    out = args.output
    config = json.loads((out / "config.json").read_text())
    load_frozen(Path(config["v4"]), config["channel_weights"], config.get("r_max", 1.98))
    protocol = apply_protocol(config.get("protocol"))
    manifest = json.loads((out / "manifest.json").read_text())
    ids = json.loads(args.ids.read_text()) if args.ids else list(range(len(manifest["cases"])))
    ids = ids[args.rank::args.shards]
    ids = [i for i in ids if not (out / "gsfit" / f"{i:06d}.npz").exists()]
    (out / "gsfit").mkdir(exist_ok=True)
    with np.load(Path(__file__).resolve().parents[1] / "assets/output_grid.npz") as env:
        r, z = env["grid_R"], env["grid_Z"]
    print(f"solve rank {args.rank}/{args.shards}: {len(ids)} frames, workers={args.workers}, "
          f"weights={config['channel_weights']}, protocol={protocol or 'v4'}", flush=True)
    started, ok, failed = time.monotonic(), 0, 0
    tag = f"{args.phase}_{args.rank}"
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("fork"),
                             initializer=_init_worker) as pool:
        futures = [pool.submit(C._gsfit_one, (str(out), i, 1, r, z)) for i in ids]
        for done, future in enumerate(as_completed(futures), 1):
            value = future.result()
            ok += int(bool(value.get("converged")))
            failed += int(not value.get("converged"))
            if done % 50 == 0 or done == len(ids):
                row = dict(rank=args.rank, shards=args.shards, done=done, total=len(ids), converged=ok,
                           failed=failed, seconds=round(time.monotonic() - started, 1))
                write_json(out / f"progress_{tag}.json", row)
                print(row, flush=True)
    write_json(out / f"done_{tag}.json", dict(frames=len(ids), converged=ok, failed=failed,
                                              seconds=time.monotonic() - started))


def pack(args):
    """与真实磁诊断实验 reconstruction/evaluate.py 的 pack 相同的 v4 补缺规则。"""
    out = args.output
    config = json.loads((out / "config.json").read_text())
    load_frozen(Path(config["v4"]), {})
    from training.gsfit_postprocess import fill_scalar

    fallback_dir = args.fallback.resolve() if args.fallback else None
    manifest = json.loads((out / "manifest.json").read_text())
    groups = {}
    for case in manifest["cases"]:
        groups.setdefault(case["official_index"], []).append(case)
    bundles, q_all, b_all, missing, used_fallback = {}, [], [], [], 0
    for i, cases in sorted(groups.items()):
        with np.load(out / "public_shots" / f"{i:04d}.npz") as z:
            times = z["efit_ms"]
        psi = [None] * len(times)
        q, b = np.full(len(times), np.nan), np.full(len(times), np.nan)
        converged = []
        for case in cases:
            path = out / "gsfit" / f"{case['id']:06d}.npz"
            if not path.exists():
                missing.append(case["id"])
                continue
            with np.load(path) as z:
                own = bool(z["converged"])
            if not own and fallback_dir is not None:
                alt = fallback_dir / "gsfit" / f"{case['id']:06d}.npz"
                if alt.exists():
                    with np.load(alt) as z:
                        if bool(z["converged"]):
                            path, own = alt, True
                            used_fallback += 1
            if not own:
                continue
            with np.load(path) as z:
                f = case["frame"]
                psi[f] = np.asarray(z["psi"], np.float32)
                q[f] = float(z["q95"]) if np.isfinite(z["q95"]) and float(z["q95"]) > 0 else np.nan
                b[f] = float(z["betaN"]) if np.isfinite(z["betaN"]) and float(z["betaN"]) >= 0 else np.nan
                converged.append(f)
        q_all.extend(q[np.isfinite(q)])
        b_all.extend(b[np.isfinite(b)])
        bundles[i] = (times, psi, q, b, converged)
    if missing and not args.allow_missing:
        raise SystemExit(f"{len(missing)} frames not solved yet (e.g. {missing[:5]})")
    fallback = {"q95": float(np.median(q_all)), "betaN": float(np.median(b_all))}
    arrays, per_shot = {}, []
    for i, (times, psi, q, b, converged) in bundles.items():
        if not any(v is not None for v in psi):
            per_shot.append(dict(index=i, frames=len(times), converged=0, missing=True))
            continue
        flux, _, _ = C._fill_shot(times, psi, q.copy(), b.copy())
        q, nq, fq = fill_scalar(times, q, fallback["q95"])
        b, nb, fb = fill_scalar(times, b, fallback["betaN"])
        for name, value in (("psirz", flux), ("q95", q), ("betaN", b)):
            assert np.isfinite(value).all(), (i, name)
            arrays[f"shot_{i:04d}_{name}"] = value
        per_shot.append(dict(index=i, frames=len(times), converged=len(converged), q95_filled=nq,
                             betaN_filled=nb, q95_fallback=fq, betaN_fallback=fb, missing=False))
    tmp = out / f"submission_mast.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    tmp.replace(out / "submission_mast.npz")
    summary = dict(shots=len(bundles), attempted=sum(len(v[0]) for v in bundles.values()),
                   converged=sum(len(v[-1]) for v in bundles.values()), unsolved=len(missing),
                   scalar_fallback=fallback, channel_weights=config["channel_weights"],
                   protocol=config.get("protocol"), fallback_run=str(fallback_dir) if fallback_dir else None,
                   fallback_frames=used_fallback, per_shot=per_shot)
    write_json(out / "pack.json", summary)
    print({k: v for k, v in summary.items() if k != "per_shot"}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["stage", "solve", "pack"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--input", type=Path, help="stage：输入目录")
    p.add_argument("--v4", type=Path, default=V4_DEFAULT, help="冻结 v4 运行目录（含 src/、fast_backend_guarded_final/）")
    p.add_argument("--channel-weight", action="append", metavar="NAME=MULT")
    p.add_argument("--rank", type=int, default=int(os.environ.get("SLURM_PROCID", 0)))
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--ids", type=Path, help="solve：只求解这些 case id（JSON 列表）")
    p.add_argument("--phase", default="full")
    p.add_argument("--allow-missing", action="store_true")
    p.add_argument("--dof", type=int, help="stage：p′/FF′ 多项式自由度；不给则沿用 v4 原版全部剖面设置")
    p.add_argument("--reg-scale", type=float, default=1.0, help="stage：正则倍率（v4 原版 1）")
    p.add_argument("--reg-form", choices=["last", "diag"], default="last", help="stage：正则矩阵形式")
    p.add_argument("--init", choices=sorted(INITS), default="v4", help="stage：初始边界 R/a/κ")
    p.add_argument("--max-iter", type=int, default=100, help="stage：GSFit 最大迭代数（v4 原版 100）")
    p.add_argument("--r-max", type=float, default=1.98, help="stage：GSFit 求解网格 R 上限（v4 原版 1.98；官方网格 2.0）")
    p.add_argument("--fallback", type=Path, help="pack：本运行未收敛帧改用该运行目录的逐帧结果")
    args = p.parse_args()
    args.output = args.output.resolve()
    if args.mode == "stage" and args.input is None:
        p.error("stage needs --input")
    {"stage": stage, "solve": solve, "pack": pack}[args.mode](args)


if __name__ == "__main__":
    main()
