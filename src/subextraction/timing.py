from __future__ import annotations

from .models import SubtitleSegment
from .srt import format_subtitle_text

MIN_GAP_SECONDS = 0.04
DROP_UNDER_SECONDS = 0.6
MAX_DURATION_MARGIN = 1.10

TIMING_RANGES = [
    (15, 1.0, 1.9),
    (25, 1.5, 2.7),
    (39, 2.4, 3.5),
    (46, 3.5, 4.2),
    (55, 3.8, 5.0),
]


def minimum_duration_for_text(text: str, floor_seconds: float) -> float:
    minimum, _ = duration_range_for_text(text, floor_seconds)
    return minimum


def duration_range_for_text(text: str, floor_seconds: float) -> tuple[float, float]:
    char_count = display_char_count(text)

    for max_chars, minimum, maximum in TIMING_RANGES:
        if char_count <= max_chars:
            return max(floor_seconds, minimum), maximum
    return max(floor_seconds, 3.8), 5.0


def display_char_count(text: str) -> int:
    display_text = format_subtitle_text(text)
    lines = [line for line in display_text.splitlines() if line.strip()]
    return len(" ".join(line.strip() for line in lines))


def apply_timing_safety(
    segments: list[SubtitleSegment],
    min_segment_seconds: float,
    gap_seconds: float = MIN_GAP_SECONDS,
    audit_events: list[dict[str, object]] | None = None,
) -> list[SubtitleSegment]:
    if not segments:
        return []

    ordered = sorted(segments, key=lambda segment: (segment.start_seconds, segment.end_seconds))
    return _adjust_once(ordered, min_segment_seconds, gap_seconds, audit_events=audit_events)


def _adjust_once(
    ordered: list[SubtitleSegment],
    min_segment_seconds: float,
    gap_seconds: float,
    audit_events: list[dict[str, object]] | None = None,
) -> list[SubtitleSegment]:
    adjusted: list[SubtitleSegment] = []

    for index, segment in enumerate(ordered):
        start = max(0.0, segment.start_seconds)
        source_end = max(start, segment.end_seconds)
        minimum, maximum = duration_range_for_text(segment.text, min_segment_seconds)
        fallback_end = start + minimum
        ceiling_end = start + maximum * MAX_DURATION_MARGIN
        desired_end = min(max(source_end, fallback_end), ceiling_end)
        if desired_end > source_end:
            _record_timing_event(
                audit_events,
                "duration_extended",
                segment,
                {
                    "original_end_seconds": source_end,
                    "adjusted_end_seconds": desired_end,
                    "minimum_duration_seconds": minimum,
                },
            )
        elif desired_end < source_end:
            _record_timing_event(
                audit_events,
                "duration_capped",
                segment,
                {
                    "original_end_seconds": source_end,
                    "adjusted_end_seconds": desired_end,
                    "maximum_duration_seconds": maximum,
                },
            )

        if index + 1 < len(ordered):
            next_start = max(0.0, ordered[index + 1].start_seconds)
            latest_non_overlap_end = max(start, next_start - gap_seconds)
            if desired_end > latest_non_overlap_end:
                non_overlap_duration = latest_non_overlap_end - start
                if non_overlap_duration >= DROP_UNDER_SECONDS:
                    _record_timing_event(
                        audit_events,
                        "trimmed_to_avoid_overlap",
                        segment,
                        {
                            "original_end_seconds": desired_end,
                            "adjusted_end_seconds": latest_non_overlap_end,
                            "next_start_seconds": next_start,
                            "duration_after_trim_seconds": non_overlap_duration,
                        },
                    )
                    desired_end = latest_non_overlap_end
                else:
                    _record_timing_event(
                        audit_events,
                        "overlap_preserved_to_keep_subtitle",
                        segment,
                        {
                            "kept_end_seconds": desired_end,
                            "next_start_seconds": next_start,
                            "non_overlap_duration_seconds": non_overlap_duration,
                            "drop_threshold_seconds": DROP_UNDER_SECONDS,
                        },
                    )

        adjusted.append(
            SubtitleSegment(
                start_seconds=start,
                end_seconds=desired_end,
                text=segment.text,
                frame_indices=segment.frame_indices,
                observation_count=segment.observation_count,
            )
        )

    return adjusted


def _record_timing_event(
    audit_events: list[dict[str, object]] | None,
    reason: str,
    segment: SubtitleSegment,
    details: dict[str, object] | None = None,
) -> None:
    if audit_events is None:
        return
    event: dict[str, object] = {
        "stage": "timing",
        "reason": reason,
        "start_seconds": segment.start_seconds,
        "end_seconds": segment.end_seconds,
        "duration_seconds": segment.end_seconds - segment.start_seconds,
        "text": segment.text,
        "frame_indices": segment.frame_indices,
        "observation_count": segment.observation_count,
    }
    if details:
        event.update(details)
    audit_events.append(event)


def remove_overlaps(segments: list[SubtitleSegment], gap_seconds: float = MIN_GAP_SECONDS) -> list[SubtitleSegment]:
    if not segments:
        return []

    adjusted = list(segments)
    for index in range(len(adjusted) - 1):
        current = adjusted[index]
        following = adjusted[index + 1]
        latest_end = max(current.start_seconds, following.start_seconds - gap_seconds)
        if current.end_seconds <= latest_end:
            continue
        adjusted[index] = SubtitleSegment(
            start_seconds=current.start_seconds,
            end_seconds=latest_end,
            text=current.text,
            frame_indices=current.frame_indices,
            observation_count=current.observation_count,
        )
    return adjusted
