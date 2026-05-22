#!/usr/bin/env bash
set -eo pipefail

LOG_FILE="${VLLM_LOG_FILE:-$HOME/vllm-qwen25vl-3b.log}"
: > "$LOG_FILE"
exec >> "$LOG_FILE" 2>&1

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate vllm

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"

exec vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.82
