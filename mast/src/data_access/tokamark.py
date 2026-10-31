"""Read local TokaMark shots without copying or modifying the shared Zarr data."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any




DEFAULT_TOKAMARK_ROOT = Path(os.environ.get("TOKAMARK_ROOT", "data/tokamark"))


@dataclass(frozen=True)
class SignalInfo:
    """Metadata from one Zarr array; reading this does not load signal values."""

    path: str
    dims: tuple[str, ...]
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: str
    attrs: dict[str, Any]
    coordinates: dict[str, str]
    is_coordinate: bool


class TokaMark:
    """Entry point for the local `<shot>.zarr` collection."""

    def __init__(self, root: str | Path | None = None) -> None:
        configured = root or os.environ.get("CROSSMAK_TOKAMARK_ROOT") or DEFAULT_TOKAMARK_ROOT
        self.root = Path(configured).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"TokaMark data directory does not exist: {self.root}")

    def shots(self) -> list[int]:
        """List local shot IDs by directory name, without opening their arrays."""
        with os.scandir(self.root) as entries:
            ids = [int(entry.name[:-5]) for entry in entries
                   if entry.is_dir()
                   and entry.name.endswith(".zarr") and entry.name[:-5].isdigit()]
        return sorted(ids)


