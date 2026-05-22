from __future__ import annotations

import re

from .models import SubtitleSegment
from .srt import choose_split_index
from .text import clean_ocr_text, is_common_instruction_noise, is_suspicious_single_observation_text

CONSECUTIVE_SPLIT_CHAR_LIMIT = 84
SPLIT_GAP_SECONDS = 0.04
DROP_UNDER_SECONDS = 0.5
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'-])")


def cleanup_segments_before_timing(
    segments: list[SubtitleSegment],
    audit_events: list[dict[str, object]] | None = None,
) -> list[SubtitleSegment]:
    cleaned: list[SubtitleSegment] = []
    for segment in segments:
        if segment.end_seconds - segment.start_seconds < DROP_UNDER_SECONDS:
            _record_segment_event(
                audit_events,
                "short_segment_preserved_before_timing",
                segment,
                {
                    "duration_seconds": segment.end_seconds - segment.start_seconds,
                    "threshold_seconds": DROP_UNDER_SECONDS,
                },
            )
        text = clean_ocr_text(segment.text)
        if not _is_valid_segment_text(text, segment.observation_count):
            _record_segment_event(
                audit_events,
                "dropped_segment_invalid_text",
                segment,
                {"cleaned_text": text},
            )
            continue
        if _is_suspicious_but_preserved(text, segment.observation_count):
            _record_segment_event(
                audit_events,
                "suspicious_single_observation_preserved",
                segment,
                {"cleaned_text": text},
            )

        normalized = SubtitleSegment(
            start_seconds=segment.start_seconds,
            end_seconds=segment.end_seconds,
            text=text,
            frame_indices=segment.frame_indices,
            observation_count=segment.observation_count,
        )
        for split_segment in split_long_segment(normalized, audit_events=audit_events):
            split_text = clean_ocr_text(split_segment.text)
            if not _is_valid_segment_text(split_text, split_segment.observation_count):
                _record_segment_event(
                    audit_events,
                    "dropped_split_segment_invalid_text",
                    split_segment,
                    {"cleaned_text": split_text},
                )
                continue
            if _is_suspicious_but_preserved(split_text, split_segment.observation_count):
                _record_segment_event(
                    audit_events,
                    "suspicious_split_segment_preserved",
                    split_segment,
                    {"cleaned_text": split_text},
                )
            cleaned.append(
                SubtitleSegment(
                    start_seconds=split_segment.start_seconds,
                    end_seconds=split_segment.end_seconds,
                    text=split_text,
                    frame_indices=split_segment.frame_indices,
                    observation_count=split_segment.observation_count,
                )
            )
    return cleaned


def split_long_segment(
    segment: SubtitleSegment,
    audit_events: list[dict[str, object]] | None = None,
) -> list[SubtitleSegment]:
    text = " ".join(line.strip() for line in segment.text.splitlines() if line.strip())
    if len(text) <= CONSECUTIVE_SPLIT_CHAR_LIMIT:
        return [segment]

    split_parts = split_text_for_consecutive_subtitles(text)
    if split_parts is None:
        return [segment]

    first_text, second_text = split_parts
    duration = max(0.0, segment.end_seconds - segment.start_seconds)
    total_chars = max(1, len(first_text) + len(second_text))
    first_ratio = len(first_text) / total_chars
    split_time = segment.start_seconds + duration * first_ratio
    split_time = min(max(split_time, segment.start_seconds), max(segment.start_seconds, segment.end_seconds - SPLIT_GAP_SECONDS))
    second_start = min(segment.end_seconds, split_time + SPLIT_GAP_SECONDS)
    if split_time - segment.start_seconds < DROP_UNDER_SECONDS or segment.end_seconds - second_start < DROP_UNDER_SECONDS:
        _record_segment_event(
            audit_events,
            "long_segment_not_split_to_preserve_short_piece",
            segment,
            {
                "first_duration_seconds": split_time - segment.start_seconds,
                "second_duration_seconds": segment.end_seconds - second_start,
                "threshold_seconds": DROP_UNDER_SECONDS,
            },
        )
        return [segment]

    return [
        SubtitleSegment(
            start_seconds=segment.start_seconds,
            end_seconds=split_time,
            text=first_text,
            frame_indices=segment.frame_indices,
            observation_count=segment.observation_count,
        ),
        SubtitleSegment(
            start_seconds=second_start,
            end_seconds=segment.end_seconds,
            text=second_text,
            frame_indices=segment.frame_indices,
            observation_count=segment.observation_count,
        ),
    ]


def _is_valid_segment_text(text: str, observation_count: int) -> bool:
    if not text or is_common_instruction_noise(text):
        return False
    return True


def _is_suspicious_but_preserved(text: str, observation_count: int) -> bool:
    return observation_count <= 1 and is_suspicious_single_observation_text(text)


def _record_segment_event(
    audit_events: list[dict[str, object]] | None,
    reason: str,
    segment: SubtitleSegment,
    details: dict[str, object] | None = None,
) -> None:
    if audit_events is None:
        return
    event: dict[str, object] = {
        "stage": "cleanup",
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


def split_text_for_consecutive_subtitles(text: str) -> tuple[str, str] | None:
    sentence_split = split_at_sentence_boundary(text)
    if sentence_split:
        return sentence_split

    split_index = choose_split_index(text)
    if split_index is None:
        return None

    first = text[:split_index].strip()
    second = text[split_index + 1 :].strip()
    if not first or not second:
        return None
    return first, second


def split_at_sentence_boundary(text: str) -> tuple[str, str] | None:
    boundaries = [match.start() for match in _SENTENCE_BOUNDARY_RE.finditer(text)]
    if not boundaries:
        return None

    midpoint = len(text) / 2
    for boundary in sorted(boundaries, key=lambda index: abs(index - midpoint)):
        first = text[: boundary + 1].strip()
        second = text[boundary + 1 :].strip()
        if first and second:
            return first, second
    return None
