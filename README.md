# subextraction

`subextraction` extracts burned-in subtitles from video by keeping the pipeline deliberately staged:

1. OpenCV reads video frames and timestamps.
2. OpenCV crops likely subtitle regions, avoiding whole-frame OCR where possible.
3. A pluggable OCR backend reads only the cropped subtitle image.
4. A deterministic merge step combines near-duplicate readings into subtitle segments.
5. The run writes debug artifacts, JSONL OCR observations, merged segment JSON, and an SRT file.

Final SRT captions are limited to two display lines and are wrapped when a line exceeds 42 characters where practical. Cleanup happens before timing: very short items are removed, common OCR/model instruction noise is filtered before merge, long captions are split only when they cannot be wrapped cleanly, and dialogue hyphens are normalized.

The preferred OCR backend is a local vLLM OpenAI-compatible server running a vision-language model such as `Qwen/Qwen2.5-VL-3B-Instruct`. Tesseract remains available as a comparison and fallback backend.

```text
video -> OpenCV frame sampling -> subtitle-region crop -> OCR backend
      -> deterministic merge -> segments.json + subtitles.srt
```

## Install

From this directory:

```powershell
python -m pip install -e .
```

For Tesseract fallback support:

```powershell
python -m pip install -e ".[tesseract]"
```

You also need the Tesseract executable installed separately if using `--ocr-backend tesseract` or `--ocr-backend both`.

## vLLM

Expected local endpoint:

```text
http://localhost:8000/v1
```

The server should expose a vision-capable model. Check with:

```powershell
Invoke-RestMethod http://localhost:8000/v1/models
```

## Manual Workflow

Open the project:

```powershell
cd C:\py_apps
code subextraction
```

Start WSL:

```powershell
wsl -d Ubuntu
```

Start vLLM in WSL:

```bash
bash /mnt/c/py_apps/subextraction/scripts/start_vllm_qwen25vl.sh
```

Logs are written to:

```bash
~/vllm-qwen25vl-3b.log
```

Open a second WSL terminal and run extraction:

```bash
bash /mnt/c/py_apps/subextraction/scripts/run_cli_wsl.sh extract "input/example_video.mpg" --run-name example_video_full --scan-stride 12 --ocr-backend vllm --vllm-timeout-seconds 180
```

Outputs are written to:

```text
output/example_video_full/
```

The main product is:

```text
output/example_video_full/subtitles.srt
```

To rebuild SRT artifacts from an existing run without running OCR again:

```bash
bash /mnt/c/py_apps/subextraction/scripts/run_cli_wsl.sh refine output/example_video_full
```

This writes `segments_refined.json`, `subtitles_refined.srt`, and `refine_refined_report.json`.

Stop vLLM:

```bash
pkill -f "vllm serve Qwen/Qwen2.5-VL-3B-Instruct"
```

## Benchmark First

Create 20-50 representative subtitle crops and OCR them:

```powershell
subextract benchmark input\example_video.mpg --limit 40 --ocr-backend vllm
```

Useful outputs:

```text
output/<run>/
  frames/
  crops/
  preprocessed/
  ocr_results.jsonl
  benchmark_review.csv
  run_config.json
```

Manually review `benchmark_review.csv` and note missed text, hallucinated text, punctuation drift, line-break issues, and repeated-frame inconsistency.

## Extract SRT

```powershell
subextract extract input\example_video.mpg --ocr-backend vllm
```

Useful outputs:

```text
output/<run>/
  frames/
  crops/
  preprocessed/
  ocr_results.jsonl
  segments.json
  subtitles.srt
  run_config.json
```

To compare vLLM against Tesseract:

```powershell
subextract benchmark input\example_video.mpg --limit 40 --ocr-backend both
```

## WSL Commands

From WSL using a working `vllm` environment:

```bash
cd /mnt/c/py_apps/subextraction
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

PYTHONPATH=src python -m subextraction.cli benchmark input/example_video.mpg --limit 40 --ocr-backend vllm
PYTHONPATH=src python -m subextraction.cli extract input/example_video.mpg --max-crops 40 --scan-stride 12 --ocr-backend vllm
```

## Development Notes

DSPy is intentionally not required by the first implementation. The OCR prompt and backend are isolated so a later DSPy module can optimize prompts or judge OCR candidates without changing frame extraction, cropping, or SRT reconstruction.

Local inputs, generated outputs, archived prototypes, and media files are ignored by default.
