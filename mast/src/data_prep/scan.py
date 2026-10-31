"""Scan shot metadata, record channel information and assign shot-level splits.

Only consolidated metadata and small channel-label arrays are read. The output
manifest is the input to ``build_cache`` and also an audit of dataset coverage:
it counts how many shots contain each configured signal and lists raw arrays
that the data config does not use.

    PYTHONPATH=src uv run --locked --offline python -m data_prep.scan \
        --config configs/data/mast_full.json --output $CACHE/manifest.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from data_access import TokaMark
from data_access.zarr_compat import open_group

from .spec import DataConfig, load_data_config


def read_array_metadata(shot_path: Path) -> dict[str, dict[str, Any]]:
    """Return {array path: {shape, dims, dtype, units}} for one shot."""
    consolidated = shot_path / ".zmetadata"
    arrays: dict[str, dict[str, Any]] = {}
    if consolidated.is_file():
        metadata = json.loads(consolidated.read_text())["metadata"]
        for key, value in metadata.items():
            if key.endswith("/.zarray"):
                path = key[:-len("/.zarray")]
                attrs = metadata.get(path + "/.zattrs", {})
                arrays[path] = {"shape": value["shape"], "dtype": value["dtype"],
                                "dims": attrs.get("_ARRAY_DIMENSIONS", []),
                                "units": attrs.get("units", "")}
        return arrays
    v3_metadata = shot_path / "zarr.json"
    if v3_metadata.is_file():
        metadata = json.loads(v3_metadata.read_text()).get("consolidated_metadata")
        if metadata is not None:
            for path, value in metadata["metadata"].items():
                if value.get("node_type") == "array":
                    dtype = value["data_type"]
                    arrays[path] = {"shape": value["shape"],
                                    "dtype": "|O" if dtype == "string" else str(np.dtype(dtype)),
                                    "dims": value.get("dimension_names") or [],
                                    "units": value.get("attributes", {}).get("units", "")}
            return arrays
    raise ValueError(f"{shot_path}: no consolidated Zarr metadata (download with data/download_fair_mast.py)")


def is_coordinate(path: str, info: dict[str, Any]) -> bool:
    return info["dims"] == [path.split("/", 1)[1]]


def channel_dim(dims: list[str]) -> str | None:
    others = [d for d in dims if not d.startswith("time")]
    return others[0] if len(others) == 1 else None


def scan_shot(db: TokaMark, shot_id: int, config: DataConfig) -> dict[str, Any]:
    path = db.root / f"{shot_id}.zarr"
    try:
        arrays = read_array_metadata(path)
    except Exception as exc:  # A broken shot must not stop the scan.
        return {"shot": shot_id, "error": f"{type(exc).__name__}: {exc}"}
    labels: dict[str, list[str]] = {}
    label_signals = [s.path for m in config.modalities for s in m.signals
                     if s.channel_mapping == "label" and s.path in arrays]
    if label_signals:
        group = open_group(path)
        for signal in label_signals:
            dim = channel_dim(arrays[signal]["dims"])
            coordinate = f"{signal.split('/', 1)[0]}/{dim}"
            if dim and coordinate in group:
                labels[signal] = [str(v) for v in np.asarray(group[coordinate][:])]
    return {"shot": shot_id, "arrays": arrays, "labels": labels,
            "has_clock": any(c in arrays for c in config.reference_clocks)}


def split_of(shot_id: int, seed: str, val_fraction: float, test_fraction: float) -> str:
    """Deterministic shot-level split that does not depend on which shots are scanned."""
    digest = hashlib.sha1(f"{seed}:{shot_id}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    if u < test_fraction:
        return "test"
    if u < test_fraction + val_fraction:
        return "val"
    return "train"


def summarize(config: DataConfig, results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    configured = set(config.signal_paths)
    signals: dict[str, dict[str, Any]] = {}
    unused: collections.Counter[str] = collections.Counter()
    for path in config.signal_paths:
        signals[path] = {"n_shots": 0, "ndim": None, "dims": None, "units": "",
                         "max_channels": 0, "shape_counts": collections.Counter(), "labels": []}
    for result in results:
        for path, info in result["arrays"].items():
            if path not in configured:
                if not is_coordinate(path, info) and not info["dtype"].startswith(("<U", "|S", "|O", "str", "object")):
                    unused[path] += 1
                continue
            entry = signals[path]
            entry["n_shots"] += 1
            entry["ndim"] = len(info["shape"])
            entry["dims"] = info["dims"]
            entry["units"] = info["units"]
            dims = info["dims"]
            non_time = tuple(n for n, d in zip(info["shape"], dims) if not d.startswith("time"))
            entry["shape_counts"][str(list(non_time))] += 1
            if entry["ndim"] == 2:
                entry["max_channels"] = max(entry["max_channels"], non_time[0] if non_time else 1)
            for label in result["labels"].get(path, []):
                if label not in entry["labels"]:
                    entry["labels"].append(label)
    for entry in signals.values():
        entry["shape_counts"] = dict(entry["shape_counts"].most_common())
    return {"signals": signals, "unconfigured_arrays": dict(unused.most_common())}


def scan_dataset(config: DataConfig, db: TokaMark, shot_ids: list[int], *,
                 val_fraction: float = 0.05, test_fraction: float = 0.05,
                 split_seed: str = "crossmak-mast-v1", threads: int = 32) -> dict[str, Any]:
    with ThreadPoolExecutor(threads) as pool:
        results = list(pool.map(lambda s: scan_shot(db, s, config), shot_ids))
    errors = [r for r in results if "error" in r]
    usable = [r for r in results if "error" not in r and r["has_clock"]]
    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for result in usable:
        splits[split_of(result["shot"], split_seed, val_fraction, test_fraction)].append(result["shot"])
    return {
        "data_config": config.name,
        "data_root": str(db.root),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "split": {"seed": split_seed, "val_fraction": val_fraction, "test_fraction": test_fraction},
        "n_scanned": len(shot_ids),
        "n_usable": len(usable),
        "errors": errors,
        "no_reference_clock": [r["shot"] for r in results if "error" not in r and not r["has_clock"]],
        "splits": splits,
        **summarize(config, usable),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="Data config JSON")
    parser.add_argument("--output", type=Path, required=True, help="manifest.json to write")
    parser.add_argument("--data-root", type=Path, help="Directory containing <shot>.zarr")
    parser.add_argument("--shots", type=int, nargs="*", help="Only these shot IDs")
    parser.add_argument("--limit", type=int, help="Only the first N shots (after --every)")
    parser.add_argument("--every", type=int, default=1, help="Take every N-th shot (quick audits)")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--split-seed", default="crossmak-mast-v1")
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()

    config = load_data_config(args.config)
    db = TokaMark(args.data_root)
    shot_ids = args.shots or db.shots()[::args.every]
    if args.limit:
        shot_ids = shot_ids[:args.limit]
    started = time.time()
    manifest = scan_dataset(config, db, shot_ids, val_fraction=args.val_fraction,
                            test_fraction=args.test_fraction, split_seed=args.split_seed,
                            threads=args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))

    print(f"Scanned {len(shot_ids)} shots in {time.time() - started:.0f} s: "
          f"{manifest['n_usable']} usable, {len(manifest['errors'])} errors")
    print({k: len(v) for k, v in manifest["splits"].items()})
    missing = [p for p, s in manifest["signals"].items() if s["n_shots"] == 0]
    if missing:
        print("Configured signals absent from every scanned shot:", ", ".join(missing))
    if manifest["unconfigured_arrays"]:
        print("Raw arrays not in the data config:", ", ".join(manifest["unconfigured_arrays"]))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
