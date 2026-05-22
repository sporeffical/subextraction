from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator

import cv2

from .artifacts import ensure_run_dirs, read_json, read_ocr_observations, write_benchmark_csv, write_json, write_jsonl
from .config import ExtractionConfig
from .crops import crop_subtitle_region
from .merge import merge_observations_with_audit
from .models import CropArtifact, CropDraft, OcrObservation
from .ocr import build_ocr_engines
from .ocr.base import OcrResult
from .preprocessing import fingerprint_diff, image_fingerprint, preprocess_for_ocr
from .srt import write_srt
from .text import clean_ocr_text, text_similarity
from .video import get_video_info, iter_sampled_frames


def benchmark_video(config: ExtractionConfig) -> Path:
    return _run(config, mode="benchmark")


def extract_video(config: ExtractionConfig) -> Path:
    return _run(config, mode="extract")


def refine_saved_run(
    run_dir: Path,
    merge_similarity: float | None = None,
    merge_gap_seconds: float | None = None,
    min_segment_seconds: float | None = None,
    hold_seconds: float | None = None,
    output_suffix: str = "refined",
) -> Path:
    run_dir = Path(run_dir)
    observations_path = run_dir / "ocr_results.jsonl"
    if not observations_path.exists():
        raise FileNotFoundError(f"Missing OCR artifact: {observations_path}")

    run_config = _read_run_config(run_dir)
    merge_similarity = _option_or_config(merge_similarity, run_config, "merge_similarity", 0.86)
    merge_gap_seconds = _option_or_config(merge_gap_seconds, run_config, "merge_gap_seconds", 4.0)
    min_segment_seconds = _option_or_config(min_segment_seconds, run_config, "min_segment_seconds", 0.65)
    hold_seconds = _option_or_config(hold_seconds, run_config, "hold_seconds", 1.2)

    observations = read_ocr_observations(observations_path)
    segments, merge_audit = merge_observations_with_audit(
        observations,
        similarity_threshold=merge_similarity,
        max_gap_seconds=merge_gap_seconds,
        min_segment_seconds=min_segment_seconds,
        hold_seconds=hold_seconds,
    )

    suffix = f"_{output_suffix}" if output_suffix else ""
    write_json(run_dir / f"segments{suffix}.json", [segment.to_json_dict() for segment in segments])
    write_srt(segments, run_dir / f"subtitles{suffix}.srt")
    write_json(run_dir / f"merge_audit{suffix}.json", merge_audit)
    write_json(
        run_dir / f"refine{suffix}_report.json",
        {
            "source_ocr_results": str(observations_path),
            "observation_count": len(observations),
            "segment_count": len(segments),
            "merge_similarity": merge_similarity,
            "merge_gap_seconds": merge_gap_seconds,
            "min_segment_seconds": min_segment_seconds,
            "hold_seconds": hold_seconds,
            "output_segments": f"segments{suffix}.json",
            "output_srt": f"subtitles{suffix}.srt",
            "output_merge_audit": f"merge_audit{suffix}.json",
        },
    )
    return run_dir


def _run(config: ExtractionConfig, mode: str) -> Path:
    video_info = get_video_info(config.video_path)
    run_dir = config.make_run_dir(mode)
    dirs = ensure_run_dirs(run_dir)
    print(
        f"{mode}: {config.video_path} "
        f"({video_info.duration_seconds / 60:.1f} min, {video_info.frame_count} frames)"
    )

    if config.limit is None:
        artifacts = collect_and_save_candidates(config, dirs)
    else:
        candidates = collect_candidates(config, limit=config.limit)
        print(f"{mode}: selected {len(candidates)} candidate crops")
        artifacts = save_candidates(candidates, dirs)

    write_json(
        run_dir / "run_config.json",
        {
            "mode": mode,
            "config": config.to_json_dict(),
            "video": video_info.to_json_dict(),
            "candidate_count": len(artifacts),
        },
    )

    observations = run_ocr(config, artifacts)
    write_jsonl(run_dir / "ocr_results.jsonl", [obs.to_json_dict() for obs in observations])
    write_benchmark_csv(run_dir / "benchmark_review.csv", benchmark_rows(observations))

    if mode == "extract":
        segments, merge_audit = merge_observations_with_audit(
            observations,
            similarity_threshold=config.merge_similarity,
            max_gap_seconds=config.merge_gap_seconds,
            min_segment_seconds=config.min_segment_seconds,
            hold_seconds=config.hold_seconds,
        )
        write_json(run_dir / "segments.json", [segment.to_json_dict() for segment in segments])
        write_srt(segments, run_dir / "subtitles.srt")
        write_json(run_dir / "merge_audit.json", merge_audit)

    return run_dir


def _read_run_config(run_dir: Path) -> dict[str, object]:
    path = run_dir / "run_config.json"
    if not path.exists():
        return {}
    data = read_json(path)
    if isinstance(data, dict):
        config = data.get("config", {})
        if isinstance(config, dict):
            return config
    return {}


def _option_or_config(value: float | None, config: dict[str, object], key: str, default: float) -> float:
    if value is not None:
        return float(value)
    try:
        return float(config.get(key, default))
    except (TypeError, ValueError):
        return default


def collect_candidates(config: ExtractionConfig, limit: int | None = None) -> list[CropDraft]:
    if limit and limit > 0:
        return collect_representative_candidates(config, limit)
    return list(iter_candidate_drafts(config))


def collect_representative_candidates(config: ExtractionConfig, limit: int) -> list[CropDraft]:
    video_info = get_video_info(config.video_path)
    start_seconds = config.start_seconds or 0.0
    end_seconds = config.end_seconds if config.end_seconds is not None else video_info.duration_seconds
    if limit == 1:
        targets = [(start_seconds + end_seconds) / 2]
    else:
        targets = [
            start_seconds + (end_seconds - start_seconds) * index / (limit - 1)
            for index in range(limit)
        ]
    selected: list[CropDraft | None] = [None] * limit
    distances = [math.inf] * limit

    for draft in iter_candidate_drafts(config):
        target_index = min(range(limit), key=lambda index: abs(draft.timestamp_seconds - targets[index]))
        distance = abs(draft.timestamp_seconds - targets[target_index])
        if distance < distances[target_index]:
            selected[target_index] = draft
            distances[target_index] = distance

    return [draft for draft in selected if draft is not None]


def iter_candidate_drafts(config: ExtractionConfig) -> Iterator[CropDraft]:
    video_info = get_video_info(config.video_path)
    previous_fingerprint = None
    last_selected_time = -math.inf

    sampled_count = 0
    kept_count = 0
    for sample in iter_sampled_frames(
        config.video_path,
        stride=config.scan_stride,
        fps=video_info.fps,
        start_seconds=config.start_seconds,
        end_seconds=config.end_seconds,
    ):
        sampled_count += 1
        if sampled_count % 1000 == 0:
            print(
                f"scan: sampled {sampled_count} frames, "
                f"kept {kept_count} crops at {sample.timestamp_seconds / 60:.1f} min"
            )
        crop, bbox, signal_score = crop_subtitle_region(
            sample.image,
            config.bottom_fraction,
            mode=config.crop_mode,
            padding_x=config.crop_padding_x,
            padding_y=config.crop_padding_y,
        )
        preprocessed = preprocess_for_ocr(crop)
        fingerprint = image_fingerprint(preprocessed)
        diff_score = fingerprint_diff(previous_fingerprint, fingerprint)
        forced = sample.timestamp_seconds - last_selected_time >= config.force_every_seconds

        if signal_score < config.min_signal:
            previous_fingerprint = fingerprint
            continue
        if diff_score < config.min_diff and not forced:
            previous_fingerprint = fingerprint
            continue

        kept_count += 1
        yield CropDraft(
            frame_index=sample.frame_index,
            timestamp_seconds=sample.timestamp_seconds,
            frame_image=sample.image,
            crop_image=crop,
            preprocessed_image=preprocessed,
            bbox=bbox,
            signal_score=signal_score,
            diff_score=diff_score,
        )
        previous_fingerprint = fingerprint
        last_selected_time = sample.timestamp_seconds


def save_candidates(candidates: list[CropDraft], dirs: dict[str, Path]) -> list[CropArtifact]:
    artifacts: list[CropArtifact] = []
    for ordinal, candidate in enumerate(candidates, start=1):
        artifacts.append(save_candidate(candidate, dirs, ordinal))
    write_json(dirs["run"] / "candidates.json", [artifact.to_json_dict() for artifact in artifacts])
    return artifacts


def collect_and_save_candidates(config: ExtractionConfig, dirs: dict[str, Path]) -> list[CropArtifact]:
    artifacts: list[CropArtifact] = []
    for ordinal, candidate in enumerate(iter_candidate_drafts(config), start=1):
        artifact = save_candidate(candidate, dirs, ordinal)
        artifacts.append(artifact)
        if ordinal % 1000 == 0:
            write_json(dirs["run"] / "candidates.json", [item.to_json_dict() for item in artifacts])
    write_json(dirs["run"] / "candidates.json", [artifact.to_json_dict() for artifact in artifacts])
    print(f"extract: selected {len(artifacts)} candidate crops")
    return artifacts


def save_candidate(candidate: CropDraft, dirs: dict[str, Path], ordinal: int) -> CropArtifact:
    stem = f"{ordinal:05d}_frame_{candidate.frame_index:07d}"
    frame_path = dirs["frames"] / f"{stem}.jpg"
    crop_path = dirs["crops"] / f"{stem}_crop.jpg"
    preprocessed_path = dirs["preprocessed"] / f"{stem}_bw.png"

    cv2.imwrite(str(frame_path), candidate.frame_image)
    cv2.imwrite(str(crop_path), candidate.crop_image)
    cv2.imwrite(str(preprocessed_path), candidate.preprocessed_image)

    return CropArtifact(
        frame_index=candidate.frame_index,
        timestamp_seconds=candidate.timestamp_seconds,
        frame_path=frame_path,
        crop_path=crop_path,
        preprocessed_path=preprocessed_path,
        bbox=candidate.bbox,
        signal_score=candidate.signal_score,
        diff_score=candidate.diff_score,
    )


def run_ocr(config: ExtractionConfig, artifacts: list[CropArtifact]) -> list[OcrObservation]:
    engines = build_ocr_engines(config)
    observations: list[OcrObservation] = []
    for index, artifact in enumerate(artifacts, start=1):
        if index == 1 or index % 25 == 0 or index == len(artifacts):
            print(f"ocr: {index}/{len(artifacts)} crops")
        results = [engine.read(artifact.crop_path, artifact.preprocessed_path) for engine in engines]
        selected_text, flags = choose_selected_text(results)
        observations.append(
            OcrObservation(
                frame_index=artifact.frame_index,
                timestamp_seconds=artifact.timestamp_seconds,
                crop_path=artifact.crop_path,
                preprocessed_path=artifact.preprocessed_path,
                selected_text=selected_text,
                results=[result.to_json_dict() for result in results],
                flags=flags,
                signal_score=artifact.signal_score,
                diff_score=artifact.diff_score,
            )
        )
    return observations


def choose_selected_text(results: list[OcrResult]) -> tuple[str, list[str]]:
    flags: list[str] = []
    by_backend = {result.backend: clean_ocr_text(result.text) for result in results}

    for result in results:
        if result.error:
            flags.append(f"{result.backend}_error")

    vllm_text = by_backend.get("vllm", "")
    tesseract_text = by_backend.get("tesseract", "")

    if vllm_text and tesseract_text:
        similarity = text_similarity(vllm_text, tesseract_text)
        if similarity < 0.65:
            flags.append("ocr_disagreement")
        return vllm_text, flags
    if vllm_text:
        return vllm_text, flags
    if tesseract_text:
        flags.append("used_tesseract_fallback")
        return tesseract_text, flags
    return "", flags


def benchmark_rows(observations: list[OcrObservation]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for observation in observations:
        result_map = {str(result.get("backend")): result for result in observation.results}
        rows.append(
            {
                "frame_index": observation.frame_index,
                "timestamp_seconds": f"{observation.timestamp_seconds:.3f}",
                "crop_path": observation.crop_path,
                "selected_text": observation.selected_text,
                "vllm_text": result_map.get("vllm", {}).get("text", ""),
                "tesseract_text": result_map.get("tesseract", {}).get("text", ""),
                "expected_text": "",
                "failure_notes": "",
                "flags": ";".join(observation.flags),
            }
        )
    return rows
