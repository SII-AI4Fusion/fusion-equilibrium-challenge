"""Pretrain the multimodal MAE on the prepared cache (single process or torchrun).

    PYTHONPATH=src uv run --locked --offline python -m training.train \
        --config configs/train/mae_v2.json --set optim.max_steps=2000

    PYTHONPATH=src uv run --locked --offline torchrun --standalone --nproc_per_node 8 \
        -m training.train --config configs/train/mae_v2.json

The run directory receives config.json, metrics.jsonl and checkpoints/{last,best}.pt.
An existing checkpoints/last.pt is resumed automatically unless --no-resume is given.
Use run.init_from to load model weights only (optimizer and step reset) for post-training.
With objective.supervised_input (e.g. D_pf_ts_ip) and supervised_targets, each step uses
the magnetics whitelist encode path instead of random MAE masking.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from model import MultiModalMAE
from model.checkpoint import FORMAT, check_format
from model.router import ModalitySpec

from .config import TrainConfig, load_train_config
from .dataset import CacheWindows
from .losses import compute_loss
from .magnetics_inputs import prepare_supervised_batch


class Runtime:
    def __init__(self, device_name: str) -> None:
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if device_name == "auto":
            device_name = f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        if self.world_size > 1:
            dist.init_process_group("nccl" if self.device.type == "cuda" else "gloo")

    @property
    def main(self) -> bool:
        return self.rank == 0

    def mean(self, values: dict[str, float]) -> dict[str, float]:
        """Average scalars over processes, ignoring NaN ("not measured on this rank").

        Keys must match on every rank; a key that is NaN everywhere stays NaN.
        """
        if self.world_size == 1 or not values:
            return values
        keys = sorted(values)
        raw = torch.tensor([values[k] for k in keys], dtype=torch.float64, device=self.device)
        finite = torch.isfinite(raw)
        stacked = torch.stack([torch.where(finite, raw, 0.0), finite.double()])
        dist.all_reduce(stacked)
        means = torch.where(stacked[1] > 0, stacked[0] / stacked[1].clamp_min(1), math.nan)
        return dict(zip(keys, means.tolist()))


    def close(self) -> None:
        if self.world_size > 1:
            dist.destroy_process_group()


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "inputs": {k: v.to(device, non_blocking=True) for k, v in batch["inputs"].items()},
        "masks": {k: v.to(device, non_blocking=True) for k, v in batch["masks"].items()},
        "t0": batch["t0"].to(device, non_blocking=True),
    }


def learning_rate(step: int, config: TrainConfig) -> float:
    optim = config.optim
    if step < optim.warmup_steps:
        return optim.lr * (step + 1) / optim.warmup_steps
    progress = (step - optim.warmup_steps) / max(1, optim.max_steps - optim.warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return optim.lr * (optim.min_lr_ratio + (1 - optim.min_lr_ratio) * cosine)


def build_optimizer(model: torch.nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    groups = [{"params": decay, "weight_decay": config.optim.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=config.optim.lr, betas=tuple(config.optim.betas))


def forever(loader: DataLoader, sampler: DistributedSampler | None) -> Iterator[dict[str, Any]]:
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parent, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_step(model: torch.nn.Module, specs: list[ModalitySpec], batch: dict[str, Any],
             config: TrainConfig, generator: torch.Generator, autocast: Any,
             layout: dict[str, Any] | None = None):
    """One forward pass and loss. With ``mask_ratio == 0`` (evaluation only) nothing is
    hidden and every available token is scored (plain reconstruction)."""
    core = model.module if isinstance(model, DistributedDataParallel) else model
    if config.objective.supervised_input:
        if layout is None:
            raise ValueError("supervised_input requires a cache layout dict")
        inputs, masks, hidden = prepare_supervised_batch(
            batch, specs, layout, config.objective.supervised_input,
            config.objective.supervised_targets, core.patches, batch["t0"].device)
    else:
        raise ValueError("objective.supervised_input must be set (the v3 model is trained supervised)")
    with autocast:
        output = model(inputs, masks, batch["t0"], hidden)
    scored = (output.hidden if (config.objective.supervised_input or config.objective.mask_ratio > 0)
              else output.available)
    terms = compute_loss(output, inputs, masks, specs, scored,
                         config.objective.modality_weights)
    return terms, output


def terms_to_floats(terms, output, names: list[str]) -> dict[str, float]:
    hidden = output.hidden.sum().item()
    available = output.available.sum().item()
    # Fraction of available tokens that were hidden.
    values = {"loss": terms.total.item(),
              "hidden_fraction": hidden / available if available else math.nan}
    for name in names:
        # NaN marks "no loss for this modality in this batch"; it is excluded when averaging.
        values[f"recon/{name}"] = terms.per_modality[name].item() if name in terms.per_modality else math.nan
    return values


def average(records: list[dict[str, float]]) -> dict[str, float]:
    keys = records[0].keys()
    out = {}
    for key in keys:
        finite = [r[key] for r in records if not math.isnan(r[key])]
        out[key] = float(np.mean(finite)) if finite else math.nan
    return out


@torch.no_grad()
def evaluate(model, specs, loader, config, autocast, runtime, max_batches,
             layout: dict[str, Any] | None = None) -> dict[str, float]:
    model.eval()
    generator = torch.Generator(runtime.device).manual_seed(1234 + runtime.rank)
    records = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        terms, output = run_step(model, specs, to_device(batch, runtime.device), config,
                                 generator, autocast, layout=layout)
        records.append(terms_to_floats(terms, output, [s.name for s in specs]))
    model.train()
    if not records:
        return {}
    return runtime.mean(average(records))


def save_checkpoint(path: Path, model, optimizer, step: int, best: float, config: TrainConfig,
                    specs: list[ModalitySpec]) -> None:
    core = model.module if isinstance(model, DistributedDataParallel) else model
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"format": FORMAT, "model": core.state_dict(),
                "optimizer": optimizer.state_dict(), "step": step,
                "best_val": best, "config": config.to_dict(),
                "modalities": [s.__dict__ for s in specs],
                "patch_seconds": core.patch_seconds, "patches": core.patches}, tmp)
    tmp.replace(path)


def train(config: TrainConfig, resume: bool = True) -> dict[str, float]:
    runtime = Runtime(config.run.device)
    seed = config.run.seed + runtime.rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    out_dir = Path(config.run.output_dir)
    checkpoint_dir = out_dir / "checkpoints"
    data = config.data

    train_set = CacheWindows.from_config(data, "train", random_offset=True)
    val_set = CacheWindows.from_config(data, "val", max_windows=data.max_val_windows,
                                       seed=config.run.seed)
    if not len(train_set):
        raise SystemExit("The training split has no windows; check the cache and min_plasma_current")
    specs = [ModalitySpec.from_dict(s) for s in train_set.specs()]
    patches = data.window_patches
    layout = json.loads((Path(data.cache_dir) / "layout.json").read_text())

    model = MultiModalMAE(specs, patches, train_set.patch_seconds, config.model_config)
    model.to(runtime.device)
    optimizer = build_optimizer(model, config)
    step, best = 0, math.inf
    last = checkpoint_dir / "last.pt"
    init_from = Path(config.run.init_from) if config.run.init_from else None
    if resume and last.is_file():
        state = torch.load(last, map_location=runtime.device, weights_only=False)
        check_format(state, last)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step, best = state["step"], state["best_val"]
        if runtime.main:
            print(f"resumed from {last} at step {step}", flush=True)
    elif init_from is not None:
        if not init_from.is_file():
            raise SystemExit(f"run.init_from not found: {init_from}")
        state = torch.load(init_from, map_location=runtime.device, weights_only=False)
        check_format(state, init_from)
        if [ModalitySpec.from_dict(s) for s in state["modalities"]] != specs:
            raise ValueError(f"Modalities in {init_from} do not match training dataset specs")
        model.load_state_dict(state["model"])
        if runtime.main:
            print(f"initialized model weights from {init_from} (ckpt step {state.get('step')}); "
                  "optimizer and run step reset", flush=True)
    if runtime.world_size > 1:
        find_unused = config.run.freeze_encoder
        model = DistributedDataParallel(
            model, device_ids=[runtime.device.index] if runtime.device.type == "cuda" else None,
            find_unused_parameters=find_unused)

    sampler = DistributedSampler(train_set, seed=config.run.seed) if runtime.world_size > 1 else None
    loader_kwargs = {"batch_size": config.optim.batch_size, "num_workers": data.num_workers,
                     "pin_memory": runtime.device.type == "cuda",
                     "persistent_workers": data.num_workers > 0}
    train_loader = DataLoader(train_set, sampler=sampler or RandomSampler(train_set),
                              drop_last=True, **loader_kwargs)
    val_sampler = DistributedSampler(val_set, shuffle=False) if runtime.world_size > 1 else None
    val_loader = DataLoader(val_set, sampler=val_sampler, shuffle=False, **loader_kwargs)
    use_bf16 = config.optim.precision == "bf16" and runtime.device.type == "cuda"
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()

    if runtime.main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(
            {**config.to_dict(), "git_commit": git_commit(), "world_size": runtime.world_size,
             "train_windows": len(train_set), "val_windows": len(val_set),
             "parameters": sum(p.numel() for p in model.parameters()),
             "signals": train_set.signals()}, indent=1))
        print(f"train windows {len(train_set)}, val windows {len(val_set)}, "
              f"parameters {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M, "
              f"world size {runtime.world_size}, start step {step}", flush=True)
    metrics_file = open(out_dir / "metrics.jsonl", "a") if runtime.main else None

    def log(record: dict[str, Any]) -> None:
        if metrics_file is not None:
            line = json.dumps({k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                               for k, v in record.items()})
            metrics_file.write(line + "\n")
            metrics_file.flush()
            print(line, flush=True)

    generator = torch.Generator(runtime.device).manual_seed(config.run.seed * 1000 + runtime.rank + step)
    started = time.time()
    window_start, records = time.time(), []
    names = [s.name for s in specs]
    model.train()
    batches = forever(train_loader, sampler)
    stop_reason = "max_steps"
    while step < config.optim.max_steps:
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, config)
        batch = to_device(next(batches), runtime.device)
        terms, output = run_step(model, specs, batch, config, generator, autocast, layout=layout)
        optimizer.zero_grad(set_to_none=True)
        terms.total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip)
        if not torch.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)  # Skip the update on a non-finite gradient.
        else:
            optimizer.step()
        step += 1
        record = terms_to_floats(terms, output, names)
        record["grad_norm"] = grad_norm.item()
        records.append(record)

        if step % config.run.log_every == 0 or step == config.optim.max_steps:
            summary = runtime.mean(average(records))
            elapsed = time.time() - window_start
            log({"split": "train", "step": step, "lr": learning_rate(step - 1, config),
                 "windows_per_s": len(records) * config.optim.batch_size * runtime.world_size / elapsed,
                 **summary})
            window_start, records = time.time(), []
        out_of_time = config.run.max_hours > 0 and runtime.any(
            (time.time() - started) / 3600 >= config.run.max_hours)
        if out_of_time:
            stop_reason = "max_hours"
        if step % config.run.eval_every == 0 or step == config.optim.max_steps or out_of_time:
            metrics = evaluate(model, specs, val_loader, config, autocast, runtime,
                               config.run.eval_batches, layout=layout)
            if metrics:
                log({"split": "val", "step": step, **metrics})
                if metrics["loss"] < best:
                    best = metrics["loss"]
                    if runtime.main:
                        save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, step, best,
                                        config, specs)
        if runtime.main and (step % config.run.save_every == 0 or step == config.optim.max_steps
                             or out_of_time):
            save_checkpoint(last, model, optimizer, step, best, config, specs)
        if out_of_time:
            break
    if metrics_file is not None:
        log({"split": "done", "step": step, "reason": stop_reason, "best_val": best,
             "hours": (time.time() - started) / 3600})
        metrics_file.close()
    runtime.close()
    return {"step": step, "best_val": best}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="Training config JSON")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="Override a config value, e.g. optim.lr=1e-4")
    parser.add_argument("--no-resume", action="store_true", help="Ignore checkpoints/last.pt")
    parser.add_argument("--init-from", type=Path, default=None,
                        help="Load model weights only from this checkpoint (sets run.init_from)")
    args = parser.parse_args()
    overrides = list(args.overrides)
    if args.init_from is not None:
        overrides.append(f"run.init_from={args.init_from}")
    config = load_train_config(args.config, overrides)
    train(config, resume=not args.no_resume)


if __name__ == "__main__":
    main()
