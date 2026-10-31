"""Convert raw TokaMark shots into a window-ready, per-modality NumPy cache.

For each shot every modality becomes one time-first ``float32`` array on a
uniform grid (see ``resample.py``) with NaN for missing data:

    <cache>/layout.json                 channel order and shapes of every modality
    <cache>/shots/<shot>/<modality>.npy [time, channel, *spatial]
    <cache>/shots/<shot>/plasma_current.npy  mean |Ip| per patch (A)
    <cache>/shots/<shot>/meta.json      written last; marks the shot as complete
    <cache>/cache_index.json            all complete shots, their split and length
    <cache>/errors.jsonl                shots or signals that failed

Existing complete shots are skipped, so an interrupted run can be restarted.

    PYTHONPATH=src uv run --locked --offline python -m data_prep.build_cache \
        --config configs/data/mast_full.json --cache $CACHE --workers 96
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from data_access import TokaMark
from data_access.zarr_compat import array_dims, open_group

from .resample import resample_time, resize_axis
from .spec import DataConfig, Layout, ModalityLayout, build_layout, load_data_config

LAYOUT_FILE = "layout.json"
INDEX_FILE = "cache_index.json"


def read_signal(group: zarr.Group, path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (values with time last, native clock, non-time dims)."""
    array = group[path]
    dims = list(array_dims(array))
    time_dims = [i for i, d in enumerate(dims) if d.startswith("time")]
    if len(time_dims) != 1:
        raise ValueError(f"{path}: expected one time dimension, got {dims}")
    axis = time_dims[0]
    clock = f"{path.split('/', 1)[0]}/{dims[axis]}"
    if clock not in group:
        raise ValueError(f"{path}: clock {clock} is missing")
    values = np.moveaxis(np.asarray(array[:], dtype=np.float64), axis, -1)
    values[~np.isfinite(values)] = np.nan
    other_dims = [d for i, d in enumerate(dims) if i != axis]
    return values, np.asarray(group[clock][:], dtype=np.float64), other_dims


def shot_grid(group: zarr.Group, config: DataConfig) -> tuple[float, int]:
    """Grid start (s) and patch count covering the reference clock.

    The grid follows the shot's own clock: patch ``k`` is centred on
    ``t0 + k * patch_seconds`` with ``t0`` the first sample of the first
    available reference clock (``magnetics/time``, whose ``t0`` equals
    ``equilibrium/time[0]``). EFIT frames, 5 ms apart from ``t0``, therefore land
    exactly on the bin centres of a one-sample-per-patch modality instead of
    being blended with their neighbours, and faster signals are resampled over
    ``[t_k - patch / 2, t_k + patch / 2)``.
    """
    for clock in config.reference_clocks:
        if clock in group:
            times = np.asarray(group[clock][:], dtype=float)
            times = times[np.isfinite(times)]
            if times.size >= 2:
                patch = config.patch_seconds
                t0 = float(times.min())
                # One patch per centre inside the clock span; the tolerance keeps
                # float round-off from dropping the last frame.
                count = int(math.floor((times.max() - t0) / patch + 1e-9)) + 1
                return t0 - patch / 2, count
    raise ValueError("No reference clock with at least two finite samples")


def build_modality(group: zarr.Group, modality: ModalityLayout, start: float,
                   n_patches: int, patch_seconds: float,
                   errors: list[dict[str, str]]) -> tuple[np.ndarray, list[str]] | None:
    """Assemble one modality as a time-first array, or None when no signal exists."""
    spp = modality.samples_per_patch
    length = n_patches * spp
    step = patch_seconds / spp
    out = np.full(modality.cache_shape(length), np.nan, dtype=np.float32)
    present: list[str] = []
    for signal in modality.signals:
        if signal.path not in group:
            continue
        try:
            values, clock, other_dims = read_signal(group, signal.path)
            if signal.valid_range is not None:
                low, high = signal.valid_range
                values[(values < low) | (values > high)] = np.nan
            if modality.kind == "timeseries":
                if values.ndim == 1:
                    values = values[None]
                if values.ndim != 2:
                    raise ValueError(f"timeseries signal must be 1D or 2D, got {values.shape}")
                if signal.channel_mapping == "label":
                    source = signal.path.split("/", 1)[0]
                    dim = other_dims[0]
                    raw_labels = [str(v) for v in group[f"{source}/{dim}"][:]]
                    lookup = dict(zip(signal.labels, signal.columns))
                    rows = [i for i, label in enumerate(raw_labels) if label in lookup]
                    columns = [lookup[raw_labels[i]] for i in rows]
                    unknown = [label for label in raw_labels if label not in lookup]
                    if unknown:
                        errors.append({"signal": signal.path,
                                       "error": f"labels not in layout, skipped: {unknown}"})
                else:
                    rows = list(range(min(values.shape[0], len(signal.columns))))
                    columns = signal.columns[:len(rows)]
                    if values.shape[0] > len(signal.columns):
                        errors.append({"signal": signal.path, "error":
                                       f"{values.shape[0]} channels > layout {len(signal.columns)}"})
                resampled = resample_time(values[rows], clock, start, step, length)
                out[:, columns] = resampled.T
            else:
                expected = len(modality.spatial) + 1
                if values.ndim != expected:
                    raise ValueError(f"{modality.kind} signal must have {expected} dims, got {values.shape}")
                for axis, size in enumerate(modality.spatial):
                    values = resize_axis(values, axis, size)
                resampled = resample_time(values, clock, start, step, length)
                out[:, signal.columns[0]] = np.moveaxis(resampled, -1, 0)
            present.append(signal.path)
        except Exception as exc:
            errors.append({"signal": signal.path, "error": f"{type(exc).__name__}: {exc}"})
    return (out, present) if present else None


def plasma_current(group: zarr.Group, config: DataConfig, start: float, n_patches: int) -> np.ndarray:
    for path in config.plasma_current_signals:
        if path in group:
            values, clock, _ = read_signal(group, path)
            ip = resample_time(np.abs(values), clock, start, config.patch_seconds, n_patches,
                               method="mean")
            if np.isfinite(ip).any():
                return ip.astype(np.float32)
    return np.full(n_patches, np.nan, dtype=np.float32)


def build_shot(shot_id: int, split: str, data_root: str, config: DataConfig, layout: Layout,
               cache: Path, overwrite: bool) -> dict[str, Any]:
    shot_dir = cache / "shots" / str(shot_id)
    meta_path = shot_dir / "meta.json"
    if meta_path.is_file() and not overwrite:
        meta = json.loads(meta_path.read_text())
        meta["split"] = split
        return {"meta": meta, "errors": [], "skipped": True}
    errors: list[dict[str, str]] = []
    try:
        group = open_group(Path(data_root) / f"{shot_id}.zarr")
        start, n_patches = shot_grid(group, config)
        tmp_dir = shot_dir.with_name(f".{shot_id}.tmp-{os.getpid()}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True)
        modalities: dict[str, Any] = {}
        for modality in layout.modalities:
            built = build_modality(group, modality, start, n_patches, layout.patch_seconds, errors)
            if built is None:
                continue
            array, present = built
            np.save(tmp_dir / f"{modality.name}.npy", array)
            modalities[modality.name] = {"signals": present,
                                         "observed_fraction": float(np.isfinite(array).mean())}
        np.save(tmp_dir / "plasma_current.npy", plasma_current(group, config, start, n_patches))
        meta = {"shot": shot_id, "split": split, "grid_start": start,
                "n_patches": n_patches, "modalities": modalities}
        (tmp_dir / "meta.json").write_text(json.dumps(meta, indent=1))
        shutil.rmtree(shot_dir, ignore_errors=True)
        tmp_dir.rename(shot_dir)
        return {"meta": meta, "errors": [dict(e, shot=shot_id) for e in errors], "skipped": False}
    except Exception as exc:
        return {"meta": None, "skipped": False,
                "errors": [{"shot": shot_id, "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(limit=3)}]}


def write_index(cache: Path, metas: list[dict[str, Any]]) -> None:
    metas = sorted(metas, key=lambda m: m["shot"])
    index = {"n_shots": len(metas),
             "shots": [{k: m[k] for k in ("shot", "split", "grid_start", "n_patches")}
                       | {"modalities": sorted(m["modalities"])} for m in metas]}
    tmp = cache / f".{INDEX_FILE}.tmp"
    tmp.write_text(json.dumps(index))
    tmp.replace(cache / INDEX_FILE)


def prepare_layout(config: DataConfig, manifest: dict[str, Any], cache: Path) -> Layout:
    layout = build_layout(config, manifest)
    path = cache / LAYOUT_FILE
    if path.is_file():
        existing = Layout.load(path)
        if existing.to_dict() != layout.to_dict():
            raise SystemExit(f"{path} differs from the layout implied by the config and manifest. "
                             "Use a new cache directory, or delete the old cache deliberately.")
    else:
        cache.mkdir(parents=True, exist_ok=True)
        layout.save(path)
    return layout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="Data config JSON")
    parser.add_argument("--cache", type=Path, required=True, help="Cache directory")
    parser.add_argument("--manifest", type=Path, help="Default: <cache>/manifest.json")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--limit", type=int, help="Only the first N shots of the selected splits")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--overwrite", action="store_true", help="Rebuild complete shots")
    args = parser.parse_args()

    config = load_data_config(args.config)
    manifest = json.loads((args.manifest or args.cache / "manifest.json").read_text())
    layout = prepare_layout(config, manifest, args.cache)
    TokaMark(manifest["data_root"])  # Fails early if the raw data root is unavailable.
    jobs = sorted((shot, split) for split in args.splits for shot in manifest["splits"][split])
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"Building {len(jobs)} shots into {args.cache} with {args.workers} workers", flush=True)

    metas: dict[int, dict[str, Any]] = {}
    index_path = args.cache / INDEX_FILE
    if index_path.is_file():
        for entry in json.loads(index_path.read_text())["shots"]:
            meta_path = args.cache / "shots" / str(entry["shot"]) / "meta.json"
            if meta_path.is_file():
                metas[entry["shot"]] = json.loads(meta_path.read_text())
    started = time.time()
    failed = 0
    with ProcessPoolExecutor(args.workers) as pool, \
            open(args.cache / "errors.jsonl", "a") as error_log:
        futures = [pool.submit(build_shot, shot, split, manifest["data_root"], config, layout,
                               args.cache, args.overwrite) for shot, split in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            for error in result["errors"]:
                error_log.write(json.dumps(error) + "\n")
            if result["meta"] is None:
                failed += 1
            else:
                metas[result["meta"]["shot"]] = result["meta"]
            if done % 50 == 0 or done == len(futures):
                error_log.flush()
                write_index(args.cache, list(metas.values()))
                rate = done / max(time.time() - started, 1e-9)
                print(f"{done}/{len(futures)} shots, {failed} failed, {rate:.2f} shots/s", flush=True)
    write_index(args.cache, list(metas.values()))
    print(f"Index has {len(metas)} complete shots: {index_path}")


if __name__ == "__main__":
    main()
