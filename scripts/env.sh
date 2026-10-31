# Sourced by the entrypoints: repository root, interpreters, no proxies, unbuffered output.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# envs/torch (Python 3.10, torch): training, v3 inference, MAST input preparation, DIII-D
TORCH_PY=${TORCH_PY:-$REPO/.venv-torch/bin/python}
# envs/gsfit (Python 3.13 + gsfit / gsfit_rs built from third_party/gsfit): every GSFit solve
GSFIT_PY=${GSFIT_PY:-$REPO/.venv-gsfit/bin/python}
# envs/data (Python 3.11, Zarr 3): FAIR-MAST cache build
DATA_PY=${DATA_PY:-$REPO/.venv-data/bin/python}
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*' no_proxy='*' PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
