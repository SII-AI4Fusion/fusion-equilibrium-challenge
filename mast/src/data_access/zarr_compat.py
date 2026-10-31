"""Read-only helpers for both legacy TokaMark and FAIR-MAST v3 stores.

The normal training environment keeps Zarr 2. FAIR-MAST v3 preparation uses a
separate Zarr 3 environment; no shared data or existing environment is changed.
"""

from pathlib import Path

import zarr


def open_group(path: Path) -> zarr.Group:
    if (path / "zarr.json").is_file() and int(zarr.__version__.split(".")[0]) < 3:
        raise RuntimeError("FAIR-MAST v3 requires the separate Zarr 3 preparation environment; "
                           "see docs/fair_mast_v3_cache.md")
    if (path / ".zmetadata").is_file():
        return zarr.open_consolidated(str(path), mode="r")
    return zarr.open_group(str(path), mode="r")


def array_dims(array: zarr.Array) -> tuple[str, ...]:
    dims = array.attrs.get("_ARRAY_DIMENSIONS")
    if dims is None:
        dims = getattr(getattr(array, "metadata", None), "dimension_names", None)
    return tuple(dims or ())


