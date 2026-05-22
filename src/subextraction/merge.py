from __future__ import annotations

from collections import Counter

from .cleanup import cleanup_segments_before_timing
from .models import OcrObservation, SubtitleSegment
from .text import clean_ocr_text, is_near_duplicate, normalize_for_compare
from .timing import apply_timing_safety


def merge_observations(
    observations: list[OcrObservation],
    similarity_threshold: float,
    max_gap_seconds: float,
    min_segment_seconds: float,
    hold_seconds: float,
) -> list[SubtitleSegment]:
    segments, _audit = merge_observations_with_audit(
        observations,
        similarity_threshold=similarity_threshold,
        max_gap_seconds=max_gap_seconds,
        min_segment_seconds=min_segment_seconds,
        hold_seconds=hold_seconds,
    )
    return segments


def merge_observations_with_audit(
    observations: list[OcrObservation],
    similarity_threshold: float,
    max_gap_seconds: float,
    min_segment_seconds: float,
    hold_seconds: float,
) -> tuple[list[SubtitleSegment], dict[str, object]]:
    audit_events: list[dict[str, object]] = []
    dropped_observations: list[dict[str, object]] = []
    blank_observation_count = 0
    usable: list[OcrObservation] = []
    for obs in observations:
        selected_text = clean_ocr_text(obs.selected_text)
        if not selected_text:
            if str(obs.selected_text or "").strip():
                dropped_observations.append(
                    {
                        "stage": "observation_cleaning",
                        "reason": "cleaned_to_empty",
                        "frame_index": obs.frame_index,
                        "timestamp_seconds": obs.timestamp_seconds,
                        "crop_path": str(obs.crop_path),
                        "preprocessed_path": str(obs.preprocessed_path),
                        "selected_text": obs.selected_text,
                        "flags": obs.flags,
                        "signal_score": obs.signal_score,
                        "diff_score": obs.diff_score,
                    }
                )
            else:
                blank_observation_count += 1
            continue
        usable.append(
            OcrObservation(
                frame_index=obs.frame_index,
                timestamp_seconds=obs.timestamp_seconds,
                crop_path=obs.crop_path,
                preprocessed_path=obs.preprocessed_path,
                selected_text=selected_text,
                results=obs.results,
                flags=obs.flags,
                signal_score=obs.signal_score,
                diff_score=obs.diff_score,
            )
        )
    usable.sort(key=lambda obs: (obs.timestamp_seconds, obs.frame_index))
    if not usable:
        return [], _build_audit(
            observations=observations,
            usable=usable,
            raw_segments=[],
            cleaned_segments=[],
            final_segments=[],
            audit_events=audit_events,
            dropped_observations=dropped_observations,
            blank_observation_count=blank_observation_count,
            similarity_threshold=similarity_threshold,
            max_gap_seconds=max_gap_seconds,
            min_segment_seconds=min_segment_seconds,
            hold_seconds=hold_seconds,
        )

    groups: list[list[OcrObservation]] = []
    current: list[OcrObservation] = [usable[0]]

    for obs in usable[1:]:
        previous = current[-1]
        gap = obs.timestamp_seconds - previous.timestamp_seconds
        same_text = is_near_duplicate(previous.selected_text, obs.selected_text, similarity_threshold)
        same_group = same_text and gap <= max_gap_seconds
        if same_group:
            current.append(obs)
            continue
        groups.append(current)
        current = [obs]
    groups.append(current)

    raw_segments: list[SubtitleSegment] = []
    for index, group in enumerate(groups):
        start = group[0].timestamp_seconds
        end = group[-1].timestamp_seconds + hold_seconds

        raw_segments.append(
            SubtitleSegment(
                start_seconds=start,
                end_seconds=end,
                text=choose_representative_text(group),
                frame_indices=[obs.frame_index for obs in group],
                observation_count=len(group),
            )
        )
    cleaned_segments = cleanup_segments_before_timing(raw_segments, audit_events=audit_events)
    final_segments = apply_timing_safety(
        cleaned_segments,
        min_segment_seconds=min_segment_seconds,
        audit_events=audit_events,
    )
    audit = _build_audit(
        observations=observations,
        usable=usable,
        raw_segments=raw_segments,
        cleaned_segments=cleaned_segments,
        final_segments=final_segments,
        audit_events=audit_events,
        dropped_observations=dropped_observations,
        blank_observation_count=blank_observation_count,
        similarity_threshold=similarity_threshold,
        max_gap_seconds=max_gap_seconds,
        min_segment_seconds=min_segment_seconds,
        hold_seconds=hold_seconds,
    )
    return final_segments, audit


def _build_audit(
    *,
    observations: list[OcrObservation],
    usable: list[OcrObservation],
    raw_segments: list[SubtitleSegment],
    cleaned_segments: list[SubtitleSegment],
    final_segments: list[SubtitleSegment],
    audit_events: list[dict[str, object]],
    dropped_observations: list[dict[str, object]],
    blank_observation_count: int,
    similarity_threshold: float,
    max_gap_seconds: float,
    min_segment_seconds: float,
    hold_seconds: float,
) -> dict[str, object]:
    event_counts = Counter(str(event.get("reason", "unknown")) for event in audit_events)
    return {
        "policy": "subtitle_first",
        "policy_notes": [
            "Text-bearing segments are preserved through timing even if overlap is needed.",
            "Timing cleanup records warnings instead of deleting segments that would become too short.",
            "Hard OCR boilerplate and empty cleaned text are still removed before merging.",
        ],
        "settings": {
            "merge_similarity": similarity_threshold,
            "merge_gap_seconds": max_gap_seconds,
            "min_segment_seconds": min_segment_seconds,
            "hold_seconds": hold_seconds,
        },
        "input_observation_count": len(observations),
        "blank_observation_count": blank_observation_count,
        "usable_observation_count": len(usable),
        "dropped_observation_count": len(dropped_observations),
        "raw_segment_count": len(raw_segments),
        "cleaned_segment_count": len(cleaned_segments),
        "final_segment_count": len(final_segments),
        "event_counts": dict(event_counts),
        "dropped_observations": dropped_observations,
        "events": audit_events,
    }


def choose_representative_text(group: list[OcrObservation]) -> str:
    texts = [obs.selected_text.strip() for obs in group if obs.selected_text.strip()]
    if not texts:
        return ""

    normalized_counts = Counter(normalize_for_compare(text) for text in texts)
    best_norm, _ = normalized_counts.most_common(1)[0]
    candidates = [text for text in texts if normalize_for_compare(text) == best_norm]
    if not candidates:
        candidates = texts
    return max(candidates, key=lambda text: (len(normalize_for_compare(text)), len(text)))
