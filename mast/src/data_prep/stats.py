"""Robust per-channel normalization statistics from the training split only.

For every cached training shot and channel, the median, interquartile range and
standard deviation of finite values are computed (profile and field channels
pool their spatial points). The statistics saved for each channel are the
median of these per-shot values across shots:

    center = median(shot medians)
    scale  = median(shot IQRs) / 1.349   (falls back to std, then 1)

The loader applies ``asinh((x - center) / scale)``, which is close to a
z-score for typical values and compresses outliers and glitches logarithmically.

    PYTHONPATH=src uv run --locked --offline python -m data_prep.stats --cache $CACHE --workers 64
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .build_cache import INDEX_FILE, LAYOUT_FILE
from .spec import Layout

STATS_FILE = "stats.json"


def shot_statistics(shot_dir: Path, modality: str, max_rows: int, seed: int) -> np.ndarray | None:
    """Return [channel, (count, median, iqr, std)] for one cached modality."""
    path = shot_dir / f"{modality}.npy"
    if not path.is_file():
        return None
    array = np.load(path, mmap_mode="r")
    length, channels = array.shape[:2]
    rng = np.random.default_rng(seed)
    rows_per_step = int(np.prod(array.shape[2:], dtype=int))
    steps = min(length, max(1, max_rows // max(rows_per_step, 1)))
    picked = np.sort(rng.choice(length, size=steps, replace=False))
    values = np.asarray(array[picked], dtype=np.float64)
    values = np.moveaxis(values, 1, 0).reshape(channels, -1)
    result = np.full((channels, 4), np.nan)
    result[:, 0] = np.isfinite(values).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        q25, q50, q75 = np.nanpercentile(values, [25, 50, 75], axis=1)
        result[:, 1] = q50
        result[:, 2] = q75 - q25
        result[:, 3] = np.nanstd(values, axis=1)
    return result


def _shot_job(args: tuple[Path, list[str], int, int]) -> dict[str, list[list[float]] | None]:
    shot_dir, modalities, max_rows, seed = args
    out = {}
    for m in modalities:
        stats = shot_statistics(shot_dir, m, max_rows, seed)
        out[m] = None if stats is None else stats.tolist()
    return out


def aggregate(per_shot: list[np.ndarray], channels: int) -> dict[str, list[float]]:
    center = np.zeros(channels)
    scale = np.ones(channels)
    shots = np.zeros(channels, dtype=int)
    if per_shot:
        stacked = np.stack(per_shot)  # [shot, channel, stat]
        for c in range(channels):
            rows = stacked[:, c]
            rows = rows[rows[:, 0] > 0]
            shots[c] = len(rows)
            if not len(rows):
                continue
            center[c] = np.nanmedian(rows[:, 1])
            iqr = np.nanmedian(rows[:, 2]) / 1.349
            std = np.nanmedian(rows[:, 3])
            tiny = 1e-12 * max(1.0, abs(center[c]))
            scale[c] = iqr if iqr > tiny else std if std > tiny else 1.0
    return {"center": center.tolist(), "scale": scale.tolist(), "n_shots": shots.tolist()}


def compute_stats(cache: Path, split: str = "train", *, max_shots: int = 3000,
                  max_rows: int = 200_000, workers: int = 1, seed: int = 0) -> dict[str, Any]:
    layout = Layout.load(cache / LAYOUT_FILE)
    index = json.loads((cache / INDEX_FILE).read_text())
    shots = [s["shot"] for s in index["shots"] if s["split"] == split]
    if not shots:
        raise ValueError(f"No cached shots in split {split!r}")
    rng = np.random.default_rng(seed)
    if len(shots) > max_shots:
        shots = sorted(rng.choice(shots, size=max_shots, replace=False).tolist())
    names = [m.name for m in layout.modalities]
    jobs = [(cache / "shots" / str(s), names, max_rows, seed + s) for s in shots]
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            results = list(pool.map(_shot_job, jobs, chunksize=4))
    else:
        results = [_shot_job(job) for job in jobs]
    stats: dict[str, Any] = {"split": split, "n_shots": len(shots),
                             "transform": "asinh((x - center) / scale)", "modalities": {}}
    for modality in layout.modalities:
        per_shot = [np.asarray(r[modality.name]) for r in results if r[modality.name] is not None]
        stats["modalities"][modality.name] = {"channels": modality.channels,
                                              **aggregate(per_shot, modality.num_channels)}
    (cache / STATS_FILE).write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-shots", type=int, default=3000,
                        help="Random subset of split shots used for statistics")
    parser.add_argument("--max-rows", type=int, default=200_000,
                        help="Values per channel and shot, sampled over time")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    started = time.time()
    stats = compute_stats(args.cache, args.split, max_shots=args.max_shots,
                          max_rows=args.max_rows, workers=args.workers, seed=args.seed)
    print(f"Statistics from {stats['n_shots']} {args.split} shots in {time.time() - started:.0f} s: "
          f"{args.cache / STATS_FILE}")
    for name, entry in stats["modalities"].items():
        empty = sum(n == 0 for n in entry["n_shots"])
        if empty:
            print(f"  {name}: {empty}/{len(entry['n_shots'])} channels have no training data")


if __name__ == "__main__":
    main()
