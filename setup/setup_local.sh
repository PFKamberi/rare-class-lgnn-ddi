#!/usr/bin/env bash
# RTX 5090 env (same as ~/Documents/thesis_ch8/.venv): torch 2.7.0+cu128,
# dgl 2.1.0 (PyPI, CPU, patched to skip graphbolt), torchdata 0.7.1, numpy<2.
# Uses only PyPI + download.pytorch.org; data.dgl.ai is not needed.
set -euo pipefail
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
cd "$(dirname "$0")/.."
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo ">> creating venv (.venv, Python 3.11; replaces an existing one)"
rm -rf .venv
uv venv --python 3.11 .venv
source .venv/bin/activate
echo ">> torch 2.7.0 (CUDA 12.8)"
uv pip install --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.7.0" "torchvision==0.22.0" "torchaudio==2.7.0"
echo ">> DGL 2.1.0 (PyPI, CPU build) and the versions it needs"
uv pip install "dgl==2.1.0" "torchdata==0.7.1" "numpy<2.0" packaging pandas pyyaml pydantic
echo ">> patching DGL to skip graphbolt"
python setup/patch_dgl_graphbolt.py
echo ">> this package (editable) + dev tools"
uv pip install "numpy<2.0" -e ".[dev]" jupyterlab
python -m ipykernel install --user --name ddi-5090 --display-name "ddi (5090)"
echo ">> done"
