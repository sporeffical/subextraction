from __future__ import annotations

import argparse
from pathlib import Path

from .config import ExtractionConfig


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "command"):
        parser.print_help()
        return 2

    if args.command == "refine":
        from .pipeline import refine_saved_run

        run_dir = refine_saved_run(
            args.run_dir,
            merge_similarity=args.merge_similarity,
            merge_gap_seconds=args.merge_gap_seconds,
            min_segment_seconds=args.min_segment_seconds,
            hold_seconds=args.hold_seconds,
            output_suffix=args.output_suffix,
        )
        print(f"Run complete: {run_dir.resolve()}")
        return 0

    config = ExtractionConfig(
        video_path=args.video,
        output_root=args.output_root,
        run_name=args.run_name,
        scan_stride=args.scan_stride,
        bottom_fraction=args.bottom_fraction,
        crop_mode=args.crop_mode,
        crop_padding_x=args.crop_padding_x,
        crop_padding_y=args.crop_padding_y,
        min_signal=args.min_signal,
        min_diff=args.min_diff,
        force_every_seconds=args.force_every_seconds,
        start_seconds=args.start_seconds,
        end_seconds=args.end_seconds,
        limit=getattr(args, "limit", None) or getattr(args, "max_crops", None),
        ocr_backend=args.ocr_backend,
        vllm_base_url=args.vllm_base_url,
        vllm_model=args.vllm_model,
        vllm_timeout_seconds=args.vllm_timeout_seconds,
        tesseract_cmd=args.tesseract_cmd,
        merge_similarity=args.merge_similarity,
        merge_gap_seconds=args.merge_gap_seconds,
        min_segment_seconds=args.min_segment_seconds,
        hold_seconds=args.hold_seconds,
    )

    if args.command == "benchmark":
        from .pipeline import benchmark_video

        run_dir = benchmark_video(config)
    else:
        from .pipeline import extract_video

        run_dir = extract_video(config)

    print(f"Run complete: {run_dir.resolve()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subextract",
        description="Extract burned-in subtitles from video using OpenCV crops and pluggable OCR.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser("benchmark", help="Extract representative crops and OCR them for manual review.")
    add_common_args(benchmark)
    benchmark.add_argument("--limit", type=int, default=40, help="Number of representative crops to OCR.")

    extract = subparsers.add_parser("extract", help="OCR candidate crops and write merged subtitles.srt.")
    add_common_args(extract)
    extract.add_argument("--max-crops", type=int, default=None, help="Optional cap for OCR crops during extraction.")

    refine = subparsers.add_parser("refine", help="Rebuild segments and SRT from an existing run's ocr_results.jsonl.")
    refine.add_argument("run_dir", type=Path, help="Existing run directory containing ocr_results.jsonl.")
    refine.add_argument(
        "--output-suffix",
        default="refined",
        help="Suffix for regenerated artifacts. Use an empty value to replace segments.json/subtitles.srt.",
    )
    refine.add_argument("--merge-similarity", type=float, default=None, help="Override saved merge similarity.")
    refine.add_argument("--merge-gap-seconds", type=float, default=None, help="Override saved merge gap seconds.")
    refine.add_argument("--min-segment-seconds", type=float, default=None, help="Override saved minimum segment seconds.")
    refine.add_argument("--hold-seconds", type=float, default=None, help="Override saved hold seconds.")

    return parser


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("video", type=Path, help="Input video path.")
    parser.add_argument("--output-root", type=Path, default=Path("output"), help="Root output directory.")
    parser.add_argument("--run-name", default=None, help="Optional output run directory name.")
    parser.add_argument("--scan-stride", type=int, default=12, help="Inspect every Nth frame.")
    parser.add_argument("--bottom-fraction", type=float, default=0.34, help="Lower-frame fraction to search for subtitles.")
    parser.add_argument(
        "--crop-mode",
        choices=["bright", "monochrome"],
        default="bright",
        help="Text mask mode. Use monochrome for low-contrast black-and-white subtitle lettering.",
    )
    parser.add_argument("--crop-padding-x", type=int, default=24, help="Horizontal pixels to pad around detected text.")
    parser.add_argument("--crop-padding-y", type=int, default=18, help="Vertical pixels to pad around detected text.")
    parser.add_argument("--min-signal", type=float, default=0.0015, help="Minimum text-like pixel signal for a crop.")
    parser.add_argument("--min-diff", type=float, default=0.030, help="Minimum crop fingerprint change to keep a frame.")
    parser.add_argument("--force-every-seconds", type=float, default=1.8, help="Keep a crop after this gap even if similar.")
    parser.add_argument("--start-seconds", type=float, default=None, help="Optional start time.")
    parser.add_argument("--end-seconds", type=float, default=None, help="Optional end time.")
    parser.add_argument(
        "--ocr-backend",
        choices=["vllm", "tesseract", "both"],
        default="vllm",
        help="OCR backend to use.",
    )
    parser.add_argument("--vllm-base-url", default="http://localhost:8000/v1", help="OpenAI-compatible vLLM base URL.")
    parser.add_argument("--vllm-model", default=None, help="Model id. Defaults to first /v1/models result.")
    parser.add_argument("--vllm-timeout-seconds", type=float, default=120.0, help="Per-image vLLM request timeout.")
    parser.add_argument("--tesseract-cmd", default=None, help="Optional path to Tesseract executable.")
    parser.add_argument("--merge-similarity", type=float, default=0.86, help="Similarity threshold for repeated subtitles.")
    parser.add_argument("--merge-gap-seconds", type=float, default=4.0, help="Max gap for near-duplicate observations.")
    parser.add_argument("--min-segment-seconds", type=float, default=0.65, help="Minimum SRT segment duration.")
    parser.add_argument("--hold-seconds", type=float, default=1.2, help="Default hold after last matching observation.")


if __name__ == "__main__":
    raise SystemExit(main())
