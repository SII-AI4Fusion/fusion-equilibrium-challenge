"""Load MultiModalMAE checkpoints written by ``training.train``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .mae import MAEConfig, MultiModalMAE
from .router import ModalitySpec

FORMAT = "crossmak-mae-v2"
# Last commit that can load v1 (MultiModalMaskedVAE) checkpoints.
LEGACY_COMMIT = "fe297b8"


def check_format(state: dict[str, Any], path: str | Path) -> None:
    if state.get("format") != FORMAT:
        raise ValueError(
            f"{path} is not a {FORMAT} checkpoint (found {state.get('format')!r}). "
            f"v1 MultiModalMaskedVAE checkpoints need the code at commit {LEGACY_COMMIT}.")


def load_model(path: str | Path, device: str | torch.device = "cpu",
               mmap: bool = False) -> tuple[MultiModalMAE, dict[str, Any]]:
    """Return the model in eval mode and the checkpoint metadata (everything but weights).

    The metadata holds ``config``, ``modalities``, ``patches``, ``patch_seconds``,
    ``step`` and ``best_val``; the optimizer state is dropped.
    """
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)
    check_format(state, path)
    specs = [ModalitySpec.from_dict(s) for s in state["modalities"]]
    model = MultiModalMAE(specs, state["patches"], state["patch_seconds"],
                          MAEConfig.from_dict(state["config"]["model"]))
    model.load_state_dict(state["model"])
    model.eval().to(device)
    meta = {k: v for k, v in state.items() if k not in ("model", "optimizer")}
    return model, meta
