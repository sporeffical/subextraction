#!/usr/bin/env bash
set -eo pipefail

cd /mnt/c/py_apps/subextraction
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate vllm

export PYTHONPATH=src
exec python -u -m subextraction.cli "$@"
