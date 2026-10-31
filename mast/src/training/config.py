"""Training configuration: JSON files plus ``--set section.key=value`` overrides.

Unknown keys are errors, so a typo in a config or override never passes silently.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from data_prep.health import DEFAULT_REASONS
from model import MAEConfig, MaskConfig


@dataclass
class DataSection:
    cache_dir: str = ""
    window_patches: int = 32
    stride_patches: int = 8
    # Keep windows whose mean |Ip| is at least this many amperes (0 keeps all).
    min_plasma_current: float = 50_000.0
    # Subset of layout modalities to train on; null means all of them.
    modalities: list[str] | None = None
    # Modalities left out of the model entirely (neither input nor target), applied
    # after ``modalities``. E.g. Mirnov/OMAHA, whose waveform inside a 5 ms patch is
    # not predictable from the other diagnostics (see docs/followups.md).
    exclude_modalities: list[str] = field(default_factory=list)
    # Raw signals (layout paths) whose channels are always treated as missing, so their
    # values are neither input nor target while the modality layout stays unchanged.
    # E.g. summary/greenwald_density, which is computed from the EFIT minor radius.
    exclude_signals: list[str] = field(default_factory=list)
    # data_prep.health audit directory; flagged shot × channel entries become missing.
    health_dir: str = ""
    health_reasons: list[str] = field(default_factory=lambda: list(DEFAULT_REASONS))
    # Modalities whose isolated spike patches from the audit are also treated as missing.
    health_spikes: list[str] = field(default_factory=lambda: ["magnetics_slow", "pf_passive"])
    # JSON file {"shots": [...]} removed from every split (competition identities).
    exclude_shots: str = ""
    # Signal path -> [lo, hi]; values outside become missing (e.g. non-physical Thomson).
    valid_ranges: dict[str, list[float]] = field(default_factory=dict)
    # Normalized values are clipped to [-clip, clip] after the asinh transform.
    clip: float = 12.0
    num_workers: int = 8
    max_val_windows: int = 4096


@dataclass
class ObjectiveSection:
    """Masked reconstruction: MSE on hidden tokens, averaged over modalities."""

    mask_ratio: float = 0.75
    mask_strategies: dict[str, float] = field(default_factory=lambda: {"random": 1.0})
    # Relative weight of each modality's loss (default 1), e.g. {"equilibrium_fields": 2.0}.
    modality_weights: dict[str, float] = field(default_factory=dict)
    # Supervised post-training: experiment name from magnetics_eval (e.g. D_pf_ts_ip).
    # Empty string keeps the random MAE masking path.
    supervised_input: str = ""
    # Modalities reconstructed under supervised_input (e.g. ["magnetics_slow"]).
    supervised_targets: list[str] = field(default_factory=list)

    @property
    def mask(self) -> MaskConfig:
        return MaskConfig(self.mask_ratio, dict(self.mask_strategies))


@dataclass
class OptimSection:
    batch_size: int = 32  # per process
    lr: float = 3e-4
    min_lr_ratio: float = 0.05
    weight_decay: float = 0.05
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    warmup_steps: int = 1000
    max_steps: int = 100_000
    grad_clip: float = 1.0
    precision: str = "bf16"  # "bf16" autocast on CUDA, or "fp32"


@dataclass
class RunSection:
    output_dir: str = ""
    seed: int = 0
    device: str = "auto"
    log_every: int = 50
    eval_every: int = 1000
    eval_batches: int = 50
    save_every: int = 1000
    # Stop cleanly (with a checkpoint) after this many hours; 0 disables it.
    max_hours: float = 0.0
    # Load model weights only from this checkpoint (optimizer/step reset). Empty skips.
    init_from: str = ""
    # Freeze encoder-side parameters (encoder, encoder_positions, codec encoders).
    freeze_encoder: bool = False


@dataclass
class TrainConfig:
    name: str = "pretrain"
    data: DataSection = field(default_factory=DataSection)
    model: dict[str, Any] = field(default_factory=dict)  # MAEConfig fields
    objective: ObjectiveSection = field(default_factory=ObjectiveSection)
    optim: OptimSection = field(default_factory=OptimSection)
    run: RunSection = field(default_factory=RunSection)

    @property
    def model_config(self) -> MAEConfig:
        return MAEConfig.from_dict(self.model)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if not self.data.cache_dir or not self.run.output_dir:
            raise ValueError("data.cache_dir and run.output_dir must be set")
        self.objective.mask  # Validates ratio and strategies.
        self.model_config  # Validates model keys.
        if bool(self.objective.supervised_input) != bool(self.objective.supervised_targets):
            raise ValueError("supervised_input and supervised_targets must be set together")
        if self.objective.supervised_input:
            from .magnetics_inputs import SUPERVISED_INPUTS
            if self.objective.supervised_input not in SUPERVISED_INPUTS:
                raise ValueError(
                    f"supervised_input must be one of {SUPERVISED_INPUTS}, "
                    f"got {self.objective.supervised_input!r}")
        elif self.objective.mask_ratio <= 0:
            raise ValueError("objective.mask_ratio must be positive for MAE training")
        if self.optim.precision not in ("bf16", "fp32"):
            raise ValueError("optim.precision must be 'bf16' or 'fp32'")


def _build(cls: type, raw: dict[str, Any], where: str) -> Any:
    known = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise KeyError(f"Unknown keys in {where}: {sorted(unknown)}")
    values = {}
    for name, value in raw.items():
        default = getattr(cls(), name)
        values[name] = _build(type(default), value, f"{where}.{name}") if is_dataclass(default) else value
    return cls(**values)


def _parse_value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    raw = copy.deepcopy(raw)
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Override must be key=value, got {item!r}")
        node = raw
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse_value(value)
    return raw


def config_from_dict(raw: dict[str, Any], overrides: list[str] = ()) -> TrainConfig:
    config = _build(TrainConfig, apply_overrides(raw, list(overrides)), "config")
    config.validate()
    return config


def load_train_config(path: str | Path, overrides: list[str] = ()) -> TrainConfig:
    return config_from_dict(json.loads(Path(path).read_text()), overrides)
