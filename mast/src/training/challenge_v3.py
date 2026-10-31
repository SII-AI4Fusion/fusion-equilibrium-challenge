"""v3 有监督模型（E_pf_tf_ts_ip → magnetics_slow + pf_passive）在比赛数据上的推理。

比赛 parquet 经 v4 ``prepare`` 得到的 ``public_shots/<i>.npz`` 提供 Ip、PF/SOL、TF 与 Thomson
剖面（R = 0.30–1.49 m，120 点，与 FAIR-MAST v3 原始网格一致）。本模块把它们换算成 v3 缓存的
通道与单位，用缓存构建时同一个 ``data_prep.resample.resample_time`` 重采样到每个窗口，再套用
训练时的输入规则：

- Thomson ``data.valid_ranges``（n_e、T_e、p_e 及核心值）之外视为缺测；
- 卡值：放电期（|Ip| ≥ 50 kA）一阶差分为 0 的比例 > 0.9 且中位幅度 > 1% 通道尺度的输入通道，整炮缺测
  （与 ``data_prep.health`` 的输入卡值规则相同）；
- 输入尖峰不屏蔽（与训练默认一致）。

通道换算（v3 ``pf_active/coil_current`` 顺序 SOL, P2IL, P2IU, P2OL, P2OU, P3L … P6U）：
parquet 各列为 kA；SOL ×1000，P2 内/外线圈按 0.6/0.4 分配 p2l/p2u，其余 ×1000。
倍率由 81 个 public∩train 炮在 v3 缓存上逐炮回归核对（P2 内外 0.59/0.41，随年代有 ±4% 漂移；
23.7k 附近少数炮 P2 内线圈断开，比赛数据无法区分）。比赛 P3 列在 v3 中 P3 未通电（≈0 A）时
仍有约 ±1 kA、与放电同步的结构，``--p3`` 可选原值、置 0 或缺测，按 ``validate`` 结果选择。
``validate`` 子命令在缓存上核对输入与模型输出。TF 用 ``tf_A``；p_e = n_e·T_e·e。
比赛数据没有 AYC 核心值（``thomson_core``）：``--core`` 取 ``mask``（缺测，默认）或 ``top5``
（剖面最大 5 点均值估计），选择依据见 target_compare 的对比。

    python -m training.challenge_v3 infer --source <v4 输入目录> --output <新目录> --checkpoint v3.pt
    python -m training.challenge_v3 validate --source <v4 public 目录> --checkpoint v3.pt --output v.json

``infer`` 在输出目录写 case_magnetics.npz（与 v2 相同的 92 路 magnetics_slow 顺序与键）、
case_passive.npz（90 路 pf_passive），并把 manifest.json、mast_static_geometry.npz、public_shots
链接到输入目录，可直接作为 ``challenge_v4_weighted.py stage --input`` 与 passive ``--source``。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

PATCH = 0.005
SPP = 16
E_CHARGE = 1.602176634e-19
MIN_IP = 50_000.0
STUCK_INPUT = 0.9
# v3 pf_active/coil_current[i] ← (public_shots 键, 倍率)
PF_SOURCES = [
    ("gs_magnetics_sol_current", 1000.0),
    ("gs_magnetics_p2l_current", 600.0),
    ("gs_magnetics_p2u_current", 600.0),
    ("gs_magnetics_p2l_current", 400.0),
    ("gs_magnetics_p2u_current", 400.0),
    ("gs_magnetics_p3l_current", 1000.0),
    ("gs_magnetics_p3u_current", 1000.0),
    ("gs_magnetics_p4l_current", 1000.0),
    ("gs_magnetics_p4u_current", 1000.0),
    ("gs_magnetics_p5l_current", 1000.0),
    ("gs_magnetics_p5u_current", 1000.0),
    ("gs_magnetics_p6l_current", 1000.0),
    ("gs_magnetics_p6u_current", 1000.0),
]
LINKS = ("manifest.json", "mast_static_geometry.npz", "public_shots")
# FAIR-MAST v3 原始 magnetics/flux_loop_channel 顺序（v3 缓存 flux_loop_flux[i] 的含义）。v2 缓存与
# GSFit 几何（mast_static_geometry.npz 的 flux_loop_channel）用另一种顺序：两者第 5–13 路不同，
# 必须按标签对齐；探针（CCBV/OBR/OBV）两套顺序一致。已用 588 个测试炮的 v2/v3 实测互相关核对（r ≥ 0.99999、增益 1.000）。
V3_FLUX_LABELS = ["CC03", "CC04", "CC05", "CC07", "CC09", "P3U/1", "P3U/4", "P4U/4", "P5U/1", "P3L/4", "P4L/1",
                  "P4L/4", "P5L/1", "P5L/4", "P6L/1"]


def v3_columns(slow_channels, flux_labels, v3_channels):
    """GSFit/v2 通道名 → v3 magnetics_slow 列：磁通环按标签，其余按同名。"""
    cols = []
    for name in slow_channels:
        if name.startswith("magnetics/flux_loop_flux["):
            label = str(flux_labels[int(name.split("[")[1].rstrip("]"))])
            name = f"magnetics/flux_loop_flux[{V3_FLUX_LABELS.index(label)}]"
        cols.append(v3_channels.index(name))
    return cols
# validate 比较的 (thomson_core, P3) 处理组合；第一个用于输入逐通道统计。


def load_context(checkpoint, device="cpu", cache_dir=None):
    from model.checkpoint import load_model
    from training.config import config_from_dict

    model, state = load_model(checkpoint, device=device, mmap=True)
    config = config_from_dict(state["config"])
    # layout.json / stats.json of the training cache (shipped next to the weights)
    cache = Path(cache_dir or config.data.cache_dir)
    layout = json.loads((cache / "layout.json").read_text())
    stats = json.loads((cache / "stats.json").read_text())["modalities"]
    mods = {m["name"]: m for m in layout["modalities"]}
    norm = {}
    for spec in model.specs:
        m = mods[spec.name]
        shape = (len(m["channels"]), *[1] * len(m.get("spatial") or []))
        norm[spec.name] = (np.asarray(stats[spec.name]["center"], np.float64).reshape(shape),
                           np.asarray(stats[spec.name]["scale"], np.float64).reshape(shape))
    ranges = {}
    for path, (lo, hi) in (config.data.valid_ranges or {}).items():
        for m in layout["modalities"]:
            for s in m["signals"]:
                if s["path"] == path and m["name"] in norm:
                    ranges.setdefault(m["name"], []).extend((c, float(lo), float(hi)) for c in s["columns"])
    return dict(model=model, state=state, config=config, cache=cache, mods=mods, norm=norm,
                ranges=ranges, clip=float(config.data.clip))


def _series(shot):
    """15 路 16 采样/块输入的原始时间序列 [C, T]（C: 13 PF/SOL、Ip、TF）与时钟。"""
    t = np.asarray(shot["mag_t"], float)
    rows = [np.asarray(shot[key], float) * scale for key, scale in PF_SOURCES]
    rows.append(np.asarray(shot["ip_A"], float))
    rows.append(np.asarray(shot["tf_A"], float))
    return t, np.vstack(rows)


def _thomson(shot):
    """[3, 120, T]：n_e、T_e、p_e = n_e·T_e·e。"""
    ne = np.asarray(shot["ne120"], float).T
    te = np.asarray(shot["te120"], float).T
    return np.asarray(shot["ts_t"], float), np.stack([ne, te, ne * te * E_CHARGE])


def stuck_channels(shot, ctx):
    """训练同款输入卡值规则：返回 15 路时间序列中整炮判为卡值的布尔向量。"""
    from data_prep.resample import resample_time

    t, rows = _series(shot)
    ok = np.isfinite(t)
    if ok.sum() < 2:
        return np.zeros(rows.shape[0], bool)
    start = float(np.nanmin(t))
    length = int((np.nanmax(t) - start) / (PATCH / SPP))
    grid = resample_time(rows, t, start, PATCH / SPP, length)
    n = length // SPP
    ip = grid[13, : n * SPP].reshape(n, SPP).mean(axis=1)
    plasma = np.repeat(np.abs(np.nan_to_num(ip)) >= MIN_IP, SPP)
    x = grid[:, : n * SPP][:, plasma]
    scales = np.concatenate([ctx["norm"]["pf_active"][1].ravel()[:13], ctx["norm"]["summary"][1].ravel()[:1],
                             ctx["norm"]["tf"][1].ravel()[:1]])
    out = np.zeros(rows.shape[0], bool)
    if x.shape[1] < 2:
        return out
    with np.errstate(invalid="ignore"):
        d = np.diff(x, axis=1)
        fin = np.isfinite(d)
        frac = np.where(fin.sum(1) > 0, ((d == 0) & fin).sum(1) / np.maximum(fin.sum(1), 1), 0.0)
        level = np.abs(np.nanmedian(x, axis=1))
    out = (frac > STUCK_INPUT) & (np.nan_to_num(level) > 0.01 * scales)
    return out


def shot_windows(shot, t0s, ctx, core="mask", p3="parquet"):
    """一个炮的多个窗口起点 → {modality: (values[B, ...], masks[B, ...])}（已归一化，时间轴在最后）。"""
    from data_prep.resample import resample_time

    patches = ctx["model"].patches
    stuck = stuck_channels(shot, ctx)
    t, rows = _series(shot)
    ts_t, prof = _thomson(shot)
    out = {}
    specs = {s.name: s for s in ctx["model"].specs}
    B = len(t0s)
    for name, spec in specs.items():
        shape = (B, spec.channels, *spec.spatial, patches * spec.samples_per_patch)
        out[name] = [np.zeros(shape, np.float32), np.zeros(shape, bool)]
    for b, t0 in enumerate(t0s):
        series = resample_time(rows, t, float(t0), PATCH / SPP, patches * SPP)  # [15, L]
        series[stuck] = np.nan
        if p3 == "mask":
            series[5:7] = np.nan
        elif p3 == "zero":
            series[5:7] = np.where(np.isfinite(series[5:7]), 0.0, np.nan)
        raw = {}
        if "pf_active" in specs:
            r = np.full((specs["pf_active"].channels, patches * SPP), np.nan)
            r[:13] = series[:13]
            raw["pf_active"] = r
        if "summary" in specs:
            r = np.full((specs["summary"].channels, patches * SPP), np.nan)
            r[0] = series[13]
            raw["summary"] = r
        if "tf" in specs:
            raw["tf"] = series[14:15]
        if "thomson_profiles" in specs:
            raw["thomson_profiles"] = resample_time(prof, ts_t, float(t0), PATCH, patches).astype(float)
        if "thomson_core" in specs and core == "top5":
            tp = raw["thomson_profiles"]
            core_v = np.full((2, patches), np.nan)
            for ch in (0, 1):
                p = tp[ch].copy()
                lo_hi = [(lo, hi) for column, lo, hi in ctx["ranges"].get("thomson_profiles", ())
                         if column == ch]
                ok = np.isfinite(p)
                for lo, hi in lo_hi:
                    with np.errstate(invalid="ignore"):
                        ok &= (p >= lo) & (p <= hi)
                p = np.where(ok, p, -np.inf)
                top = np.sort(p, axis=0)[-5:]
                good = np.isfinite(top).all(axis=0)
                core_v[ch] = np.where(good, np.where(np.isfinite(top), top, 0).mean(axis=0), np.nan)
            raw["thomson_core"] = core_v
        for name, value in raw.items():
            center, scale = ctx["norm"][name]
            # profiles: [C, R, L]；timeseries: [C, L]
            c = center.reshape(center.shape + (1,))
            s = scale.reshape(scale.shape + (1,))
            obs = np.isfinite(value)
            for column, lo, hi in ctx["ranges"].get(name, ()):
                with np.errstate(invalid="ignore"):
                    obs[column] &= (value[column] >= lo) & (value[column] <= hi)
            z = np.arcsinh((np.nan_to_num(value) - c) / s)
            z = np.clip(np.where(obs, z, 0.0), -ctx["clip"], ctx["clip"])
            out[name][0][b] = z
            out[name][1][b] = obs
    return {k: (v[0], v[1]) for k, v in out.items()}


def run_model(ctx, windows, t0s, device):
    """返回 {target: 物理值 [B, C, L]}。与 v2 推理相同：白名单输入编码，解码器查询全部位置。"""
    import torch
    from training.magnetics_eval import predict
    from training.magnetics_inputs import input_channels

    model = ctx["model"]
    objective = ctx["config"].objective
    layout = {"modalities": list(ctx["mods"].values())}
    selected = input_channels(layout, objective.supervised_input)
    batch = {"inputs": {k: torch.from_numpy(v[0]) for k, v in windows.items()},
             "masks": {k: torch.from_numpy(v[1]) for k, v in windows.items()},
             "t0": torch.tensor(np.asarray(t0s, np.float32))}
    ctxm = (torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and ctx["config"].optim.precision == "bf16" else _null())
    with torch.inference_mode(), ctxm:
        reconstruction, _ = predict(model, batch, model.specs, selected, device)
    result = {}
    for t in objective.supervised_targets:
        z = reconstruction[t].float().cpu().numpy().astype(np.float64)
        c, s = ctx["norm"][t]
        result[t] = c.reshape(1, -1, 1) + s.reshape(1, -1, 1) * np.sinh(np.clip(z, -ctx["clip"], ctx["clip"]))
    return result


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def at_time(phys, t0s, times, spp=SPP):
    """[B, C, L] 在每个窗口的 time_s 处线性插值 → [B, C]（与 v4 infer 相同的取值方式）。"""
    step = PATCH / spp
    out = np.empty(phys.shape[:2])
    for j in range(phys.shape[0]):
        pos = (times[j] - t0s[j]) / step - 0.5
        lo = int(np.clip(np.floor(pos), 0, phys.shape[-1] - 2))
        a = float(pos - lo)
        out[j] = (1 - a) * phys[j, :, lo] + a * phys[j, :, lo + 1]
    return out


def infer(args):
    import torch

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cpu":
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    ctx = load_context(args.checkpoint, device=device, cache_dir=args.cache_dir)
    ctx["model"].to(device).eval().requires_grad_(False)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    for name in LINKS:
        link = out / name
        try:  # shards on several GPUs create the same links concurrently
            link.symlink_to((args.source / name).resolve())
        except FileExistsError:
            pass
    manifest = json.loads((out / "manifest.json").read_text())
    with np.load(out / "mast_static_geometry.npz", allow_pickle=False) as g:
        slow_channels = [str(x) for x in g["slow_channels"]]
        flux_labels = [str(x) for x in g["flux_loop_channel"]]
    mag3 = ctx["mods"]["magnetics_slow"]["channels"]
    col = v3_columns(slow_channels, flux_labels, mag3)
    pas = ctx["mods"]["pf_passive"]["channels"]
    cases = manifest["cases"]
    if args.limit_shots:
        keep = sorted({c["official_index"] for c in cases})[: args.limit_shots]
        cases = [c for c in cases if c["official_index"] in set(keep)]
    if args.shards > 1:
        keep = set(sorted({c["official_index"] for c in cases})[args.rank::args.shards])
        cases = [c for c in cases if c["official_index"] in keep]
    by_shot = {}
    for c in cases:
        by_shot.setdefault(int(c["official_index"]), []).append(c)
    n_all = len(manifest["cases"])
    pred = np.full((n_all, len(slow_channels)), np.nan, np.float32)
    ppred = np.full((n_all, len(pas)), np.nan, np.float32)
    ip = np.full(n_all, np.nan)
    started = time.monotonic()
    done = 0
    stuck_log = {}
    for i, shot_cases in sorted(by_shot.items()):
        with np.load(out / "public_shots" / f"{i:04d}.npz", allow_pickle=False) as z:
            shot = {k: z[k] for k in z.files}
        st = stuck_channels(shot, ctx)
        if st.any():
            stuck_log[i] = np.flatnonzero(st).tolist()
        for first in range(0, len(shot_cases), args.batch_size):
            part = shot_cases[first:first + args.batch_size]
            t0s = [float(c["t0"]) for c in part]
            times = [float(c["time_s"]) for c in part]
            w = shot_windows(shot, t0s, ctx, core=args.core, p3=args.p3)
            phys = run_model(ctx, w, t0s, device)
            ids = [int(c["id"]) for c in part]
            pred[ids] = at_time(phys["magnetics_slow"], t0s, times)[:, col]
            ppred[ids] = at_time(phys["pf_passive"], t0s, times)
            mt = np.asarray(shot["mag_t"], float)
            ip[ids] = np.interp(times, mt, np.asarray(shot["ip_A"], float))
        done += len(shot_cases)
        if done // 2000 != (done - len(shot_cases)) // 2000 or i == max(by_shot):
            print(f"v3 {done}/{len(cases)} cases, {time.monotonic() - started:.0f}s", flush=True)
    meta = dict(checkpoint=str(args.checkpoint), step=int(ctx["state"]["step"]), core=args.core, p3=args.p3,
                supervised_input=ctx["config"].objective.supervised_input, source=str(args.source.resolve()),
                stuck_shots=stuck_log, seconds=time.monotonic() - started, cases=len(cases))
    if args.shards > 1:
        part = out / f"v3_part{args.rank}.npz"
        np.savez_compressed(part.with_suffix(".tmp.npz"), prediction=pred, passive=ppred, ip=ip,
                            channels=np.array(slow_channels), passive_channels=np.array(pas),
                            step=int(ctx["state"]["step"]), meta=json.dumps(meta))
        part.with_suffix(".tmp.npz").replace(part)
        print(f"wrote shard {args.rank}/{args.shards}")
        return
    write_outputs(out, pred, ppred, ip, slow_channels, pas, int(ctx["state"]["step"]), meta)


def merge(args):
    out = args.output
    parts = [np.load(out / f"v3_part{r}.npz") for r in range(args.shards)]
    pred = parts[0]["prediction"].copy()
    ppred = parts[0]["passive"].copy()
    ip = parts[0]["ip"].copy()
    metas = []
    for p in parts:
        ok = np.isfinite(p["prediction"]).any(axis=1)
        pred[ok], ppred[ok], ip[ok] = p["prediction"][ok], p["passive"][ok], p["ip"][ok]
        metas.append(json.loads(str(p["meta"])))
    if not np.isfinite(pred).all() or not np.isfinite(ppred).all():
        raise SystemExit(f"merged predictions incomplete: {int((~np.isfinite(pred).all(1)).sum())} cases")
    meta = dict(metas[0], cases=sum(m["cases"] for m in metas), seconds=max(m["seconds"] for m in metas),
                stuck_shots={k: v for m in metas for k, v in m["stuck_shots"].items()}, shards=args.shards)
    write_outputs(out, pred, ppred, ip, parts[0]["channels"].tolist(), parts[0]["passive_channels"].tolist(),
                  int(parts[0]["step"]), meta)


def write_outputs(out, pred, ppred, ip, slow_channels, pas, step, meta):
    n_all = len(pred)
    tmp = out / "case_magnetics.tmp.npz"
    np.savez_compressed(tmp, ids=np.arange(n_all, dtype=np.int32), prediction=pred, ip=ip,
                        channels=np.array(slow_channels), checkpoint_step=step,
                        seconds=meta["seconds"])
    tmp.replace(out / "case_magnetics.npz")
    tmp = out / "case_passive.tmp.npz"
    np.savez_compressed(tmp, ids=np.arange(n_all, dtype=np.int32), prediction=ppred, channels=np.array(pas))
    tmp.replace(out / "case_passive.npz")
    (out / "v3_inference.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "stuck_shots"}), f"stuck shots={len(meta['stuck_shots'])}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["infer", "merge"])
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, help="directory with the training cache layout.json and stats.json "
                   "(default: the cache_dir stored in the checkpoint config)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--core", choices=["mask", "top5"], default="mask")
    p.add_argument("--p3", choices=["parquet", "zero", "mask"], default="parquet",
                   help="P3 线圈：parquet 列 ×1000 / 置 0 / 缺测（比赛 P3 列含 v3 中不存在的约 ±1 kA 结构）")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit-shots", type=int, default=0, help="infer：只算前 N 炮（冒烟）")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    args = p.parse_args()
    {"infer": infer, "merge": merge}[args.mode](args)


if __name__ == "__main__":
    main()
