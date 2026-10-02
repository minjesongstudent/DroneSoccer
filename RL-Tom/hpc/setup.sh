#!/bin/bash
# One-off: create the Python environment on the HPC. Run from the repo root.
#   bash hpc/setup.sh
# The system Python is 3.6, too old, so uv fetches its own Python into ~/.local.
set -euo pipefail

if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python 3.11 .venv
# CPU-only PyTorch: the network is too small to benefit from a GPU, and the
# default Linux wheel downloads ~2 GB of CUDA libraries.
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv gymnasium numpy opencv-python-headless stable-baselines3 tensorboard matplotlib pytest
.venv/bin/python -m pytest -q
