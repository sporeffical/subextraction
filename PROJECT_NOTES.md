# Project Notes

## Operating Principles

- vLLM vision reads cropped subtitle regions from selected frames after OpenCV isolates the subtitle area.
- Does not ask the model to inspect whole video frames with busy backgrounds and infer timing perfectly.
- OpenCV handles video decoding, frame selection, crop generation, timestamps, and debug images.
- OCR backends read cropped subtitle images only.
- A deterministic subtitle-merging module reconstructs final subtitle segments and writes SRT output.
- Keeps Tesseract available for comparison and fallback. Vision LLMs can hallucinate; classic OCR is less smart but often more literal.
- Do not trust vLLM blindly as the only OCR signal. Benchmark it on the sample video before relying on it.

## First Practical Test

1. Extract 20-50 representative subtitle crops.
2. Send each crop to the vLLM model with a strict prompt.
3. Compare outputs against expected subtitle text manually.
4. Measure failure types:
   - missed text
   - hallucinated text
   - punctuation drift
   - line-break issues
   - repeated-frame inconsistency

## OCR Prompt

```text
Read only the subtitle text visible in this image.
Return an empty string if there is no subtitle.
Do not describe the image.
Do not guess missing words.
Preserve punctuation and line breaks when clear.
```

## Current Architecture

```text
video -> sampled frames -> subtitle crop -> preprocessing variants
      -> OCR backend(s) -> normalized observations
      -> deterministic merge -> segments.json + subtitles.srt
```

The OCR backend is intentionally pluggable. Current backends:

- `vllm`: local OpenAI-compatible vLLM vision endpoint.
- `tesseract`: optional local Tesseract executable.
- `both`: run both backends and preserve both outputs for comparison.

## Runtime Artifacts

Each run writes a timestamped output directory containing:

- `frames/`: selected full frames for inspection.
- `crops/`: cropped subtitle regions sent to OCR.
- `preprocessed/`: high-contrast crops used for signal scoring and Tesseract fallback.
- `ocr_results.jsonl`: one OCR observation per crop.
- `benchmark_review.csv`: manual review sheet for benchmark runs.
- `segments.json`: merged subtitle segments for extraction runs.
- `subtitles.srt`: final subtitle file for extraction runs. Final SRT layout wraps subtitles over two lines when a caption exceeds 42 characters, but never creates more than two subtitle lines.
- `subtitles_refined.srt`: optional regenerated SRT from saved `ocr_results.jsonl` when using the `refine` command.
- `run_config.json`: reproducibility details.

## Timing Safety

Detected subtitle timing should follow the source observations as much as possible while staying close to the configured ranges:

- Under 0.5 seconds: remove before timing adjustment. Final processed subtitles under 0.6 seconds are removed after no-overlap clamping.
- 1-15 characters, including spaces: 1.0-1.9 seconds.
- 16-25 characters: 1.5-2.7 seconds.
- 26-39 characters: 2.4-3.5 seconds.
- 40-46 characters: 3.5-4.2 seconds.
- 47-55 characters: 3.8-5.0 seconds.
- Long captions are wrapped to two display lines first. Consecutive subtitle splitting is reserved for captions over roughly 84 characters that cannot be kept readable as one subtitle.

These are safety ranges, not blind target durations. Durations are capped to the range maximum plus a 10 percent margin. Minimums are extended where possible, but if extending a subtitle would overlap the next subtitle, no-overlap takes priority and the subtitle is clamped before the next start time.

## Cleanup Rules

Cleanup happens before timing safety and is applied both before merge and after any long-caption split:

- Remove very short subtitles.
- Remove common OCR/model instruction noise, including lines beginning with `Please do not`, `This is a test`, `The quick brown fox`, prompt echoes, placeholder/test/demo/preview phrases, and common stock vLLM completions.
- Remove subtitle status and self-reference messages such as `Subtitles are not available`, `Subtitles are optional`, `Subtitles are not supported`, `Subtitle text here`, and generated language-menu outputs such as `Subtitles: English, French...`.
- Preserve real subtitle credit lines, such as translator names or copyright notices.
- Normalize leading dialogue hyphens so `-Where? -I know.` becomes two speaker lines.
- Prefer sentence boundaries when splitting display lines.
