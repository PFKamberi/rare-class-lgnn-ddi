#!/usr/bin/env bash
# Colab install. Run from a notebook cell:   !bash /content/ddi-linegraph/setup/colab_setup.sh
#
# DGL's last wheels support torch <= 2.4 and Python <= 3.12. The default Colab
# runtime (2026.10) ships Python 3.13, where these wheels do not exist.
# Pin the runtime first: Runtime > Change runtime type > Runtime version 2026.07
# (Python 3.12). Pinned versions stay available for one year after release.
set -euo pipefail
PYV=$(python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
if [ "$PYV" != "3.12" ] && [ "$PYV" != "3.11" ] && [ "$PYV" != "3.10" ]; then
  echo "!! Python $PYV detected. DGL needs Python <= 3.12."
  echo "!! Runtime > Change runtime type > Runtime version: 2026.07, then rerun."
  exit 1
fi
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# Same pins as the thesis notebooks (torch 2.4 cu118 runs on T4, L4 and A100).
pip install -q torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
pip install -q dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/cu118/repo.html
pip install -q -e "$REPO_DIR"
python "$REPO_DIR/setup/verify_gpu.py"
