from __future__ import annotations

import re
from pathlib import Path

from .models import SubtitleSegment

MAX_SUBTITLE_LINE_CHARS = 42
MAX_SUBTITLE_LINES = 2
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'-])")


def format_srt_time(seconds: float) -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def format_subtitle_text(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.strip().splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return ""

    sentence_lines = split_sentences_to_two_lines(" ".join(lines))
    if sentence_lines and all(len(line) <= MAX_SUBTITLE_LINE_CHARS for line in sentence_lines):
        return "\n".join(sentence_lines)

    if len(lines) <= MAX_SUBTITLE_LINES and all(len(line) <= MAX_SUBTITLE_LINE_CHARS for line in lines):
        return "\n".join(lines)

    if len(lines) == MAX_SUBTITLE_LINES and any(line.startswith("-") for line in lines):
        return "\n".join(wrap_lines("\n".join(lines), MAX_SUBTITLE_LINE_CHARS))

    split_text = split_to_two_lines(" ".join(lines))
    if all(len(line) <= MAX_SUBTITLE_LINE_CHARS for line in split_text.splitlines()):
        return split_text

    return "\n".join(wrap_lines(split_text, MAX_SUBTITLE_LINE_CHARS))


def split_sentences_to_two_lines(text: str) -> list[str] | None:
    parts = [part.strip() for part in _SENTENCE_BOUNDARY_RE.split(text) if part.strip()]
    if len(parts) < 2:
        return None

    if len(parts) == 2:
        return parts

    best: list[str] | None = None
    best_balance: int | None = None
    for index in range(1, len(parts)):
        candidate = [" ".join(parts[:index]), " ".join(parts[index:])]
        if any(len(line) > MAX_SUBTITLE_LINE_CHARS for line in candidate):
            continue
        balance = abs(len(candidate[0]) - len(candidate[1]))
        if best is None or best_balance is None or balance < best_balance:
            best = candidate
            best_balance = balance
    return best


def split_to_two_lines(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= MAX_SUBTITLE_LINE_CHARS:
        return text

    split_at = choose_split_index(text)
    if split_at is None:
        return text

    first = text[:split_at].strip()
    second = text[split_at + 1 :].strip()
    return f"{first}\n{second}" if second else first


def wrap_lines(text: str, max_chars: int) -> list[str]:
    wrapped: list[str] = []
    for source_line in text.splitlines():
        current = ""
        for word in source_line.split():
            word_parts = split_long_word(word, max_chars)
            if len(word_parts) > 1:
                if current:
                    wrapped.append(current)
                    current = ""
                wrapped.extend(word_parts[:-1])
                current = word_parts[-1]
                continue
            if not current:
                current = word
                continue
            candidate = f"{current} {word}"
            if len(candidate) <= max_chars:
                current = candidate
            else:
                wrapped.append(current)
                current = word
        if current:
            wrapped.append(current)
    return wrapped


def split_long_word(word: str, max_chars: int) -> list[str]:
    if len(word) <= max_chars:
        return [word]
    return [word[index : index + max_chars] for index in range(0, len(word), max_chars)]


def choose_split_index(text: str) -> int | None:
    spaces = [index for index, char in enumerate(text) if char == " "]
    if not spaces:
        return None

    midpoint = len(text) / 2

    def score(index: int) -> tuple[int, float]:
        first_len = index
        second_len = len(text) - index - 1
        overflow = max(0, first_len - MAX_SUBTITLE_LINE_CHARS) + max(0, second_len - MAX_SUBTITLE_LINE_CHARS)
        balance = abs(index - midpoint)
        return overflow, balance

    return min(spaces, key=score)


def write_srt(segments: list[SubtitleSegment], output_path: Path) -> None:
    lines: list[str] = []
    for index, segment in enumerate(segments, start=1):
        lines.extend(
            [
                str(index),
                f"{format_srt_time(segment.start_seconds)} --> {format_srt_time(segment.end_seconds)}",
                format_subtitle_text(segment.text),
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")
