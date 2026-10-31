"""Data configuration and the cache layout derived from it.

The data config (``configs/data/*.json``) names every raw signal and the
modality it belongs to. ``scan`` adds per-signal channel information from the
real dataset; ``build_layout`` combines both into the fixed tensor layout that
the cache, the normalization statistics and the model all share.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

KINDS = ("timeseries", "profile", "field")


@dataclass(frozen=True)
class SignalSpec:
    path: str
    # "index": raw channel i goes to column i (labels are renamed between MAST
    # eras, e.g. XSX_HCAMU#1 vs /xsx/HCAM/U/1). "label": columns follow the
    # union of channel labels found by scan (for channel sets that really differ).
    channel_mapping: str = "index"
    # Values outside [low, high] are fill values or glitches and become missing.
    valid_range: tuple[float, float] | None = None


@dataclass(frozen=True)
class ModalityConfig:
    name: str
    kind: str
    samples_per_patch: int
    signals: tuple[SignalSpec, ...]
    # Spatial size after resampling: [R] for profiles, [H, W] for fields.
    size: tuple[int, ...] = ()


@dataclass(frozen=True)
class DataConfig:
    name: str
    patch_seconds: float
    reference_clocks: tuple[str, ...]
    plasma_current_signals: tuple[str, ...]
    modalities: tuple[ModalityConfig, ...]

    @property
    def signal_paths(self) -> list[str]:
        return [s.path for m in self.modalities for s in m.signals]


def load_data_config(path: str | Path) -> DataConfig:
    raw = json.loads(Path(path).read_text())
    modalities = []
    for item in raw["modalities"]:
        kind = item["kind"]
        if kind not in KINDS:
            raise ValueError(f"Modality {item['name']!r}: kind must be one of {KINDS}")
        spp = int(item["samples_per_patch"])
        if spp < 1 or spp & (spp - 1):
            raise ValueError(f"Modality {item['name']!r}: samples_per_patch must be a power of two")
        size = tuple(int(n) for n in item.get("size", ()))
        if kind == "profile" and len(size) != 1 or kind == "field" and len(size) != 2:
            raise ValueError(f"Modality {item['name']!r}: profile needs size [R], field needs [H, W]")
        signals = tuple(SignalSpec(s["path"], s.get("channel_mapping", "index"),
                                   tuple(s["valid_range"]) if "valid_range" in s else None)
                        for s in item["signals"])
        for s in signals:
            if s.valid_range is not None and (len(s.valid_range) != 2
                                              or s.valid_range[0] >= s.valid_range[1]):
                raise ValueError(f"{s.path}: valid_range must be [low, high] with low < high")
            if s.channel_mapping not in ("index", "label"):
                raise ValueError(f"{s.path}: channel_mapping must be 'index' or 'label'")
            if s.channel_mapping == "label" and kind != "timeseries":
                raise ValueError(f"{s.path}: label mapping is only defined for timeseries channels")
        modalities.append(ModalityConfig(item["name"], kind, spp, signals, size))
    names = [m.name for m in modalities]
    if len(set(names)) != len(names):
        raise ValueError("Modality names must be unique")
    paths = [s.path for m in modalities for s in m.signals]
    if len(set(paths)) != len(paths):
        raise ValueError("Each signal may belong to only one modality")
    return DataConfig(
        name=raw["name"],
        patch_seconds=float(raw["patch_seconds"]),
        reference_clocks=tuple(raw["reference_clocks"]),
        plasma_current_signals=tuple(raw["plasma_current_signals"]),
        modalities=tuple(modalities),
    )


@dataclass
class SignalLayout:
    """Where one raw signal lands inside its modality tensor."""

    path: str
    channel_mapping: str
    # timeseries: output column of each raw channel (index or label order).
    # profile/field: the single quantity index this signal occupies.
    columns: list[int]
    labels: list[str] = field(default_factory=list)
    valid_range: list[float] | None = None


@dataclass
class ModalityLayout:
    name: str
    kind: str
    samples_per_patch: int
    channels: list[str]
    spatial: list[int]
    signals: list[SignalLayout]

    @property
    def num_channels(self) -> int:
        return len(self.channels)

    def cache_shape(self, length: int) -> tuple[int, ...]:
        """Cache arrays are time-first so a window is one contiguous slice."""
        return (length, self.num_channels, *self.spatial)

    def model_spec(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "channels": self.num_channels,
                "samples_per_patch": self.samples_per_patch, "spatial": list(self.spatial)}


@dataclass
class Layout:
    data_config: str
    patch_seconds: float
    modalities: list[ModalityLayout]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Layout:
        modalities = [ModalityLayout(**{**m, "signals": [SignalLayout(**s) for s in m["signals"]]})
                      for m in raw["modalities"]]
        return cls(raw["data_config"], raw["patch_seconds"], modalities)

    @classmethod
    def load(cls, path: str | Path) -> Layout:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1, ensure_ascii=False))

    def get(self, name: str) -> ModalityLayout:
        for modality in self.modalities:
            if modality.name == name:
                return modality
        raise KeyError(f"Modality {name!r} is not in the layout")


def valid_range(signal: SignalSpec) -> list[float] | None:
    return list(signal.valid_range) if signal.valid_range is not None else None


def build_layout(config: DataConfig, manifest: dict[str, Any]) -> Layout:
    """Fix channel counts and orders from the scanned manifest."""
    signal_info = manifest["signals"]
    modalities = []
    for m in config.modalities:
        channels: list[str] = []
        signals: list[SignalLayout] = []
        for quantity, s in enumerate(m.signals):
            info = signal_info.get(s.path)
            if info is None or info["n_shots"] == 0:
                raise ValueError(f"{s.path} was not found in any scanned shot; remove it from "
                                 f"the data config or rescan more shots")
            if m.kind != "timeseries":
                channels.append(s.path)
                signals.append(SignalLayout(s.path, s.channel_mapping, [quantity],
                                            valid_range=valid_range(s)))
                continue
            if info["ndim"] == 1:
                labels = [""]
            elif s.channel_mapping == "label":
                labels = list(info["labels"])
            else:
                labels = [str(i) for i in range(info["max_channels"])]
            start = len(channels)
            channels.extend(s.path if label == "" else f"{s.path}[{label}]" for label in labels)
            signals.append(SignalLayout(s.path, s.channel_mapping,
                                        list(range(start, len(channels))),
                                        labels if s.channel_mapping == "label" else [],
                                        valid_range(s)))
        modalities.append(ModalityLayout(m.name, m.kind, m.samples_per_patch, channels,
                                         list(m.size), signals))
    return Layout(config.name, config.patch_seconds, modalities)
