"""PyTorch dataset of fixed-length, multimodal windows read from the prepared cache."""

from __future__ import annotations

import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from data_prep.build_cache import INDEX_FILE, LAYOUT_FILE
from data_prep.health import DEFAULT_REASONS, HEALTH_FILE, REASONS, SPIKES_FILE
from data_prep.spec import Layout, ModalityLayout
from data_prep.stats import STATS_FILE


class CacheWindows(Dataset):
    """Windows of ``window_patches`` patches from every shot of one split.

    Each item is a dict with

    - ``inputs[modality]``: float32 ``[C, *spatial, L]``, normalized, 0 where missing
    - ``masks[modality]``: bool of the same shape, True where a value was observed
    - ``t0``: window start time (s), ``shot`` and ``start_patch``

    With ``random_offset`` every access shifts the window start by a random
    amount below ``stride_patches``, so training sees all start positions.

    Channels of ``exclude_signals`` stay in the tensors but are always missing
    (value 0, mask False), so the model never sees them and they are never scored.

    ``health`` points to a ``data_prep.health`` audit directory. Shot × channel entries
    flagged with any of ``health_reasons``, and the spike patches of ``health_spikes``
    modalities, are treated as missing in the same way. ``valid_ranges`` maps signal paths
    to ``[lo, hi]``; values outside are missing. ``exclude_shots`` removes shots entirely
    (e.g. competition test identities) before any window is built.
    """

    def __init__(self, cache_dir: str | Path, split: str, *, window_patches: int,
                 stride_patches: int, min_plasma_current: float = 0.0,
                 modalities: list[str] | None = None,
                 exclude_modalities: list[str] | tuple[str, ...] = (),
                 exclude_signals: list[str] | tuple[str, ...] = (), clip: float = 12.0,
                 random_offset: bool = False, max_windows: int | None = None,
                 seed: int = 0, health: str | Path | None = None,
                 health_reasons: list[str] | tuple[str, ...] = DEFAULT_REASONS,
                 health_spikes: list[str] | tuple[str, ...] = ("magnetics_slow", "pf_passive"),
                 exclude_shots: list[int] | tuple[int, ...] | set[int] = (),
                 valid_ranges: dict[str, list[float]] | None = None) -> None:
        self.cache = Path(cache_dir)
        self.layout = Layout.load(self.cache / LAYOUT_FILE)
        names = modalities or [m.name for m in self.layout.modalities]
        for name in exclude_modalities:
            self.layout.get(name)  # A typo must not silently keep the modality.
        names = [n for n in names if n not in set(exclude_modalities)]
        if not names:
            raise ValueError("No modalities left after exclude_modalities")
        self.modalities: list[ModalityLayout] = [self.layout.get(n) for n in names]
        self.exclude_signals = set(exclude_signals)
        unknown = self.exclude_signals - {s.path for m in self.layout.modalities for s in m.signals}
        if unknown:
            raise KeyError(f"exclude_signals not in {LAYOUT_FILE}: {sorted(unknown)}")
        self.blanked: dict[str, list[int]] = {}
        for m in self.modalities:
            columns = sorted(c for s in m.signals if s.path in self.exclude_signals for c in s.columns)
            if columns:
                self.blanked[m.name] = columns
        self.window_patches = window_patches
        self.stride_patches = stride_patches
        self.clip = clip
        self.random_offset = random_offset
        self.patch_seconds = self.layout.patch_seconds

        stats = json.loads((self.cache / STATS_FILE).read_text())["modalities"]
        self.center: dict[str, np.ndarray] = {}
        self.scale: dict[str, np.ndarray] = {}
        for m in self.modalities:
            if stats[m.name]["channels"] != m.channels:
                raise ValueError(f"{STATS_FILE} does not match {LAYOUT_FILE} for {m.name}")
            shape = (m.num_channels, *[1] * len(m.spatial))
            self.center[m.name] = np.asarray(stats[m.name]["center"], np.float32).reshape(shape)
            self.scale[m.name] = np.asarray(stats[m.name]["scale"], np.float32).reshape(shape)

        self.ranges: dict[str, list[tuple[int, float, float]]] = {}
        for path, (lo, hi) in (valid_ranges or {}).items():
            matched = [(m, s) for m in self.modalities for s in m.signals if s.path == path]
            if not matched:
                raise KeyError(f"valid_ranges signal not in the selected modalities: {path}")
            for m, s in matched:
                self.ranges.setdefault(m.name, []).extend((c, float(lo), float(hi)) for c in s.columns)

        index = json.loads((self.cache / INDEX_FILE).read_text())
        excluded = {int(s) for s in exclude_shots}
        self.shots = [s for s in index["shots"] if s["split"] == split and int(s["shot"]) not in excluded]
        self.health_bad, self.health_spikes = self._load_health(health, health_reasons, health_spikes)
        windows = []
        for i, shot in enumerate(self.shots):
            n = shot["n_patches"]
            if n < window_patches:
                continue
            starts = np.arange(0, n - window_patches + 1, stride_patches)
            if min_plasma_current > 0:
                ip = np.load(self.cache / "shots" / str(shot["shot"]) / "plasma_current.npy")
                ip = np.nan_to_num(ip, nan=0.0)
                total = np.concatenate([[0.0], np.cumsum(ip)])
                mean = (total[starts + window_patches] - total[starts]) / window_patches
                starts = starts[mean >= min_plasma_current]
            windows.extend((i, int(s)) for s in starts)
        if max_windows is not None and len(windows) > max_windows:
            rng = np.random.default_rng(seed)
            picked = np.sort(rng.choice(len(windows), size=max_windows, replace=False))
            windows = [windows[j] for j in picked]
        self.windows = windows
        self._arrays: OrderedDict[Path, np.ndarray] = OrderedDict()

    def _load_health(self, health, reasons, spike_modalities):
        """Per shot: {modality: bad columns} and {modality: [(patch, column), ...]}."""
        if not health:
            return {}, {}
        health = Path(health)
        unknown = set(reasons) - set(REASONS)
        if unknown:
            raise KeyError(f"Unknown health reasons: {sorted(unknown)}")
        bits = sum(REASONS[r] for r in reasons)
        wanted = {int(s["shot"]) for s in self.shots}
        names = {m.name: m for m in self.modalities}
        bad: dict[int, dict[str, np.ndarray]] = {}
        with np.load(health / HEALTH_FILE) as z:
            shots = z["shots"].astype(int)
            for name, m in names.items():
                if f"reason_{name}" not in z.files:
                    continue
                if list(z[f"channels_{name}"]) != list(m.channels):
                    raise ValueError(f"{HEALTH_FILE} channels do not match {LAYOUT_FILE} for {name}")
                flagged = (z[f"reason_{name}"] & bits) > 0
                for i in np.nonzero(flagged.any(axis=1))[0]:
                    if shots[i] in wanted:
                        bad.setdefault(int(shots[i]), {})[name] = np.nonzero(flagged[i])[0]
        spikes: dict[int, dict[str, np.ndarray]] = {}
        with np.load(health / SPIKES_FILE) as z:
            shots = z["shots"].astype(int)
            for name in spike_modalities:
                if name not in names:
                    continue
                rows = z[name]
                for i in np.unique(rows[:, 0]) if len(rows) else []:
                    if shots[i] in wanted:
                        spikes.setdefault(int(shots[i]), {})[name] = rows[rows[:, 0] == i, 1:]
        return bad, spikes

    @classmethod
    def from_config(cls, data: Any, split: str, **kwargs: Any) -> "CacheWindows":
        """Build from a ``training.config.DataSection``; kwargs override (e.g. max_windows)."""
        exclude = ()
        if data.exclude_shots:
            exclude = json.loads(Path(data.exclude_shots).read_text())["shots"]
        options = dict(window_patches=data.window_patches, stride_patches=data.stride_patches,
                       min_plasma_current=data.min_plasma_current, modalities=data.modalities,
                       exclude_modalities=data.exclude_modalities,
                       exclude_signals=data.exclude_signals, clip=data.clip,
                       health=data.health_dir or None, health_reasons=data.health_reasons,
                       health_spikes=data.health_spikes, exclude_shots=exclude,
                       valid_ranges=data.valid_ranges)
        options.update(kwargs)
        return cls(data.cache_dir, split, **options)

    def __len__(self) -> int:
        return len(self.windows)

    def specs(self) -> list[dict[str, Any]]:
        return [m.model_spec() for m in self.modalities]

    def signals(self) -> dict[str, list[str]]:
        """Raw signal paths whose values can reach the model, per modality."""
        return {m.name: [s.path for s in m.signals if s.path not in self.exclude_signals]
                for m in self.modalities}

    def _array(self, path: Path) -> np.ndarray:
        array = self._arrays.pop(path, None)
        if array is None:
            array = np.load(path, mmap_mode="r")
        self._arrays[path] = array
        while len(self._arrays) > 256:
            self._arrays.popitem(last=False)
        return array

    def __getitem__(self, item: int) -> dict[str, Any]:
        shot_index, start = self.windows[item]
        shot = self.shots[shot_index]
        if self.random_offset and self.stride_patches > 1:
            last = shot["n_patches"] - self.window_patches
            start = min(start + random.randrange(self.stride_patches), last)
        shot_dir = self.cache / "shots" / str(shot["shot"])
        bad = self.health_bad.get(int(shot["shot"]), {})
        spikes = self.health_spikes.get(int(shot["shot"]), {})
        inputs, masks = {}, {}
        for m in self.modalities:
            spp = m.samples_per_patch
            length = self.window_patches * spp
            if m.name in shot["modalities"]:
                raw = np.asarray(self._array(shot_dir / f"{m.name}.npy")[start * spp:start * spp + length],
                                 dtype=np.float32)
                observed = np.isfinite(raw)
                if m.name in self.blanked:
                    observed[:, self.blanked[m.name]] = False
                if m.name in bad:
                    observed[:, bad[m.name]] = False
                for column, lo, hi in self.ranges.get(m.name, ()):
                    with np.errstate(invalid="ignore"):
                        observed[:, column] &= (raw[:, column] >= lo) & (raw[:, column] <= hi)
                if m.name in spikes:
                    rows = spikes[m.name]
                    rows = rows[(rows[:, 0] >= start) & (rows[:, 0] < start + self.window_patches)]
                    for patch, column in rows:
                        p = (patch - start) * spp
                        observed[p:p + spp, column] = False
                values = np.arcsinh((raw - self.center[m.name]) / self.scale[m.name])
                values = np.clip(np.where(observed, values, 0.0), -self.clip, self.clip)
                # Cache is time-first; the model expects time last.
                values = np.moveaxis(values, 0, -1)
                observed = np.moveaxis(observed, 0, -1)
            else:
                shape = (m.num_channels, *m.spatial, length)
                values = np.zeros(shape, np.float32)
                observed = np.zeros(shape, bool)
            inputs[m.name] = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))
            masks[m.name] = torch.from_numpy(np.ascontiguousarray(observed))
        t0 = shot["grid_start"] + start * self.patch_seconds
        return {"inputs": inputs, "masks": masks, "t0": torch.tensor(t0, dtype=torch.float32),
                "shot": shot["shot"], "start_patch": start}


