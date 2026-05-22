#!/usr/bin/env bash
set -eo pipefail

cd /mnt/c/py_apps/subextraction

if ! curl -fsS http://127.0.0.1:8000/v1/models >/dev/null; then
  echo "vLLM is not ready at http://127.0.0.1:8000/v1/models" >&2
  exit 1
fi

common_args=(
  --scan-stride "${SCAN_STRIDE:-12}"
  --bottom-fraction "${BOTTOM_FRACTION:-0.50}"
  --crop-padding-x "${CROP_PADDING_X:-32}"
  --crop-padding-y "${CROP_PADDING_Y:-28}"
  --ocr-backend vllm
  --vllm-timeout-seconds "${VLLM_TIMEOUT_SECONDS:-180}"
)

run_extract() {
  local input_file="$1"
  local run_name="$2"

  echo "START ${input_file} -> ${run_name} $(date -Is)"
  bash scripts/run_cli_wsl.sh extract "input/${input_file}" --run-name "${run_name}" "${common_args[@]}"
  echo "DONE ${input_file} -> ${run_name} $(date -Is)"
}

echo "batch_started $(date -Is)"

run_extract "example_video_01.mpg" "example_video_01_full"
run_extract "example_video_02.mpg" "example_video_02_full"

echo "batch_complete $(date -Is)"
