from __future__ import annotations

import re
from difflib import SequenceMatcher


_FENCE_RE = re.compile(r"^```(?:text)?|```$", re.IGNORECASE | re.MULTILINE)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_SPACE_RE = re.compile(r"[ \t]+")
_COMPARE_RE = re.compile(r"[^a-z0-9]+")
_LONG_DIGIT_RUN_RE = re.compile(r"\d{25,}")
_LEADING_DIALOGUE_DASH_RE = re.compile(r"^\s*[-\u2013\u2014]\s*")
_INLINE_DIALOGUE_DASH_RE = re.compile(r"\s+[-\u2013\u2014]\s*")
_LANGUAGE_NAMES = {
    "arabic",
    "bulgarian",
    "chinese",
    "croatian",
    "czech",
    "danish",
    "dutch",
    "english",
    "estonian",
    "faroese",
    "finnish",
    "french",
    "german",
    "greek",
    "greenlandic",
    "hebrew",
    "hungarian",
    "icelandic",
    "indonesian",
    "italian",
    "japanese",
    "korean",
    "latvian",
    "lithuanian",
    "mandarin",
    "norwegian",
    "polish",
    "portuguese",
    "romanian",
    "russian",
    "simplified",
    "slovak",
    "slovenian",
    "spanish",
    "swedish",
    "thai",
    "traditional",
    "turkish",
    "ukrainian",
    "vietnamese",
}


def clean_ocr_text(text: str | None) -> str:
    if not text:
        return ""
    cleaned = _FENCE_RE.sub("", str(text))
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _CONTROL_RE.sub("", cleaned)
    raw_lines = [_SPACE_RE.sub(" ", line).strip() for line in cleaned.split("\n")]
    lines: list[str] = []
    for line in raw_lines:
        if not line:
            continue
        lines.extend(_split_dialogue_line(line))
    cleaned_text = "\n".join(lines).strip().strip('"').strip("'").strip()
    cleaned_text = _strip_descriptive_prefix(cleaned_text)
    cleaned_text = _strip_descriptive_suffix(cleaned_text)
    cleaned_text = cleaned_text.strip("*").strip()
    if _looks_like_prompt_leak(cleaned_text):
        return ""
    if is_common_instruction_noise(cleaned_text):
        return ""
    if _is_empty_subtitle_response(cleaned_text):
        return ""
    return cleaned_text


def _split_dialogue_line(line: str) -> list[str]:
    normalized = _LEADING_DIALOGUE_DASH_RE.sub("- ", line, count=1)
    if not normalized.startswith("- "):
        return [normalized]

    body = normalized[2:].strip()
    parts = [part.strip() for part in _INLINE_DIALOGUE_DASH_RE.split(body) if part.strip()]
    if len(parts) <= 1:
        return [normalized]
    return [f"- {part}" for part in parts]


def _strip_descriptive_prefix(text: str) -> str:
    prefixes = (
        "here is the text as it appears:",
        "here is the subtitle text from the image:",
        "overlaid subtitle/intertitle text:",
        "plaintext ",
        "the text appears to be in russian and reads:",
        "the text reads:",
        "the subtitles visible in the image are:",
        "the subtitle text visible in the image is:",
        "the subtitle text is:",
        "subtitle text:",
        "visible subtitle text:",
    )
    lower = text.lower().lstrip()
    for prefix in prefixes:
        if lower.startswith(prefix):
            return text[len(prefix) :].strip().strip('"').strip("'").strip()
    return text


def _strip_descriptive_suffix(text: str) -> str:
    suffix_markers = (
        "there are no subtitles or intertitles present in this image",
        "there is no subtitle or intertitle present in this image",
        "no subtitles or intertitles present in this image",
        "no subtitle or intertitle present in this image",
    )
    lower = text.lower()
    for marker in suffix_markers:
        index = lower.find(marker)
        if index > 0:
            return text[:index].strip().strip(".").strip("-").strip()
    return text


def _looks_like_prompt_leak(text: str) -> bool:
    lower = text.lower()
    if lower.startswith("this is a request to read only the subtitle text"):
        return True
    prompt_markers = (
        "read only the subtitle text visible in this image",
        "this is a request to read only the subtitle text",
        "return an empty string if there is no subtitle",
        "do not describe the image",
        "do not guess missing words",
        "preserve punctuation and line breaks",
        "preserved punctuation and line breaks",
    )
    marker_count = sum(1 for marker in prompt_markers if marker in lower)
    if marker_count >= 2:
        return True
    return marker_count == 1 and len(lower) <= 80


def _is_empty_subtitle_response(text: str) -> bool:
    normalized = _SPACE_RE.sub(" ", text.strip().lower()).strip().strip(".")
    exact_empty = {"empty string", '""', "''", "none", "no subtitle", "no subtitles"}
    if normalized in exact_empty:
        return True
    empty_phrases = (
        "no subtitle text visible",
        "no subtitle visible",
        "no subtitles visible",
        "there is no subtitle",
        "there are no subtitles",
        "no visible subtitle",
        "subtitle text is not clear enough to read",
        "subtitle text is not present in the image",
        "subtitle text is not present",
        "subtitles are optional",
        "subtitles are available for all videos",
        "subtitles are not supported",
        "subtitles are not available",
        "subtitles are not provided",
        "no additional subtitle text",
    )
    return len(normalized) < 90 and any(phrase in normalized for phrase in empty_phrases)


def is_common_instruction_noise(text: str) -> bool:
    normalized = normalize_raw_for_compare(text)
    if not normalized:
        return not any(char.isalnum() for char in text)

    exact_noise = {
        "the",
        "subtitle",
        "subtitles",
        "subtitle text",
        "subtitle text here",
        "subtitle this is a subtitle",
        "subtitle this is a test subtitle",
        "subtitle text visible in this image",
        "subtitles by name",
        "subtitles by subtitleguy",
        "subtitles by subtitleguy com",
        "subtitled by name of subtitler",
        "the subtitle text visible in this image",
        "the subtitle text visible in the image",
        "subtitles the subtitle text visible in this image",
        "visible in the image is the subtitle text visible in the image is",
        "the subtitle text",
        "the subtitle text is here",
        "the subtitle text is not clear enough to be read",
        "the end",
        "do not",
        "happy birthday",
        "please sit down",
        "subtitles none",
        "subtitle none",
        "no subtitle",
        "no subtitles",
        "subtitles removed for review",
        "subtitles removed for privacy",
        "subtitles removed for editing purposes only",
        "subtitles removed for editing purposes",
        "not available",
        "no subtitle or intertitle",
        "no subtitle or intertitle detected",
        "no subtitle or intertitle present",
        "no subtitle or intertitle visible",
        "no subtitle or intertitle text visible",
        "no subtitle or intertitle text present",
        "no subtitles or intertitles detected",
        "no subtitles or intertitles present",
        "overlapped subtitle text not available",
        "the text is not clear enough to read",
        "please wait for the next scene",
        "0123456789",
        "007",
        "527",
    }
    if normalized in exact_noise:
        return True
    if normalized.startswith("subtitle ") and normalized.removeprefix("subtitle ").replace(" ", "").isdigit():
        return True

    if normalized.startswith("please do not "):
        return True
    if normalized.startswith("please note ") and any(
        marker in normalized
        for marker in (
            "unable to read the text",
            "contains no subtitles",
            "contains no subtitle",
            "no subtitles or intertitles",
            "no subtitle or intertitle",
        )
    ):
        return True
    if normalized.startswith("please provide the image"):
        return True
    if normalized.startswith("please provide a clearer image"):
        return True
    if normalized.startswith("i need to read "):
        return True
    if normalized.startswith("please note ") and any(
        marker in normalized
        for marker in (
            "test",
            "placeholder",
            "demonstration",
            "sample image",
            "preview",
            "final version",
            "medical scan",
            "information provided",
            "image may contain copyrighted material",
            "copyrighted material",
        )
    ):
        return True
    if normalized.startswith("this is a test"):
        return True
    if normalized.startswith("the quick brown fox"):
        return True
    if normalized.startswith("the text is not clearly legible"):
        return True
    if _LONG_DIGIT_RUN_RE.search(str(text)):
        return True
    if "no other text visible in the image" in normalized:
        return True
    if normalized.startswith("the text appears to be in russian and contains names"):
        return True
    if normalized.startswith("the text appears to be in russian") and "without additional context" in normalized:
        return True
    if normalized.startswith("the text appears to be in a language that uses a non latin script"):
        return True
    if "happy birthday" in normalized and any(
        marker in normalized for marker in ("cheerleaders", "holding", "sign", "written")
    ):
        return True
    if "for demonstration purposes only" in normalized:
        return True
    if "test image" in normalized or "placeholder" in normalized:
        return True

    subtitle_status_phrases = (
        "subtitles are not available",
        "subtitles not available",
        "subtitles are unavailable",
        "subtitles are available",
        "subtitles are optional",
        "subtitles are not supported",
        "subtitles not supported",
        "subtitles and intertitles are not present",
        "subtitle text is not clear",
        "subtitle text is unclear",
        "subtitles on",
        "subtitles off",
        "subtitle on",
        "subtitle off",
    )
    if any(normalized.startswith(phrase) for phrase in subtitle_status_phrases):
        return True

    if _looks_like_subtitle_language_menu(normalized):
        return True
    if _looks_like_image_description_or_refusal(normalized):
        return True
    if _looks_like_stock_model_completion(normalized):
        return True
    if _looks_like_scene_description_noise(normalized):
        return True

    return False


def is_suspicious_single_observation_text(text: str) -> bool:
    """Catch stock vLLM completions that usually appear on blank one-off crops."""
    normalized = normalize_raw_for_compare(text)
    if not normalized:
        return True
    if is_common_instruction_noise(text):
        return True
    return _looks_like_stock_model_completion(normalized) or _looks_like_single_observation_scene_description(normalized)


def _looks_like_subtitle_language_menu(normalized: str) -> bool:
    if not normalized.startswith("subtitles "):
        return False
    words = set(normalized.split())
    language_hits = len(words & _LANGUAGE_NAMES)
    return language_hits >= 4


def _looks_like_image_description_or_refusal(normalized: str) -> bool:
    starters = (
        "the image",
        "the subtitle text visible in the image is",
        "this image",
        "the picture",
        "the photo",
        "visible in the image is",
        "there is no image",
        "there is no visible",
        "there are no visible",
        "therefore i cannot",
        "i need to read",
        "i cannot read",
        "i cannot provide",
        "i cannot extract",
        "i ve read the text in the image",
        "i have read the text in the image",
        "i read the text in the image",
        "cannot read",
        "cannot extract",
        "unable to read",
        "unable to extract",
        "i m sorry but i can t read",
        "i m sorry but i cannot read",
        "i m sorry but i am unable to read",
        "i can t read the entire image",
        "i cannot read the entire image",
    )
    if normalized.startswith(starters):
        return True

    description_markers = (
        "visible text",
        "visible subtitle",
        "subtitles in the image",
        "subtitle text from this image",
        "subtitle text in the image",
        "subtitle of the image",
        "no additional subtitle text",
        "no other text present in the image",
        "there is no further text",
        "text present in the image",
        "visible part of the subtitle",
        "visible part of the text",
        "provided image",
        "is visible in the image",
        "blurry and unclear",
        "blurred and unclear",
        "image you provided",
        "information or context provided in the image",
        "medical scan",
    )
    return any(marker in normalized for marker in description_markers)


def _looks_like_stock_model_completion(normalized: str) -> bool:
    stock_starts = (
        "the world is a book",
        "the world s largest",
        "the world s most",
        "the worlds largest",
        "the worlds most",
        "the president s speech",
        "the president s car arrives",
        "the president of the united states",
        "the presidents speech",
        "the piano",
        "the sun s rays",
        "the window s reflection",
        "the window bars cast shadows",
        "the wall creating a mesmerizing",
        "the wall creating a pattern",
        "the hollywood walk of fame",
        "the trapeze marvel of the age",
        "and presentation of tea often accompanied",
        "once upon a time in a land",
        "on july 20 1969",
        "his friend who had been his companion",
        "man who had been his companion",
        "and the man who had been with him",
        "and the man who had been",
        "xxl xxxl",
        "only the bride s dress is visible",
        "the snow falls",
        "it is the largest natural satellite",
        "it is the largest of the four natural satellites",
        "snowy landscape",
        "chessboard with pieces",
        "natural satellite",
        "curiosity rover",
        "the knowledge of the mind is the key",
        "the man is performing",
        "the man is playing",
        "the woman is performing",
        "the woman is playing",
        "a man is performing",
        "a man is playing",
        "a woman is performing",
        "a woman is playing",
        "the american flag is displayed",
        "president obama s speech",
        "president obama signs",
        "pointing sword at lock",
        "sleeping baby",
        "clap your hands",
        "made with love",
        "the cd you are now listening to is a preview of the full album",
    )
    if normalized.startswith(stock_starts):
        return True
    if "translates to" in normalized:
        return True
    if normalized.count("who had been") >= 4:
        return True
    tokens = normalized.split()
    if len(tokens) >= 6 and set(tokens) <= {"xl", "xxl", "xxxl"}:
        return True
    if _has_repeated_ngram_loop(tokens):
        return True
    compact = normalized.replace(" ", "")
    return len(compact) >= 20 and compact.isdigit()


def _looks_like_single_observation_scene_description(normalized: str) -> bool:
    starts = (
        "cheerleaders performing",
        "cheerleaders holding",
        "the 1950s era cheerleaders",
        "hand holding",
        "the hand of a person",
        "close up of",
        "close up ",
        "with the focus on",
        "the dress is made of",
        "snowflakes fall",
        "snowflakes falling",
        "please stand in a straight line",
        "please stand at",
        "please stand by",
        "please stand for",
        "please stand up",
    )
    if normalized.startswith(starts):
        return True
    if "shown in close up" in normalized and "background is blurred" in normalized:
        return True
    return _looks_like_scene_description_noise(normalized) or normalized in {"100 polyester", "100 percent polyester"}


def _looks_like_scene_description_noise(normalized: str) -> bool:
    starts = (
        "beautiful woman in a carriage",
        "candles lit in a serene setting",
        "candles lit hands holding them",
        "hand on face dark background",
        "hand with a black background",
        "the table is set for a formal dinner",
        "the table is set for dinner",
        "the table is set for a meal",
        "with the skateboarder s feet clearly visible",
        "jumping off the roof",
        "the bucket is being tilted",
        "ground likely to pour something into it",
        "jesus said to his disciples",
        "belongs to such as these",
        "camel in the desert",
        "camel",
        "kite flying on the beach",
        "ants marching on the sand",
        "int spacecraft night",
        "a figure stands at the control panel",
        "the background is a dark star filled",
        "with distant lights streaking",
    )
    if normalized.startswith(starts):
        return True
    if normalized.startswith("die bombenangriffe") and "katastrophalen zerst" in normalized:
        return True
    return "soft glow illuminating the scene" in normalized


def _has_repeated_ngram_loop(tokens: list[str]) -> bool:
    if len(tokens) < 10:
        return False
    for width in range(2, 6):
        counts: dict[tuple[str, ...], int] = {}
        for index in range(len(tokens) - width + 1):
            ngram = tuple(tokens[index : index + width])
            counts[ngram] = counts.get(ngram, 0) + 1
        if not counts:
            continue
        phrase, count = max(counts.items(), key=lambda item: item[1])
        if count >= 4 and count * len(phrase) >= len(tokens) * 0.55:
            return True
    return False


def normalize_for_compare(text: str) -> str:
    return normalize_raw_for_compare(clean_ocr_text(text))


def normalize_raw_for_compare(text: str) -> str:
    text = str(text).lower()
    text = text.replace("|", "i")
    text = _COMPARE_RE.sub(" ", text)
    return " ".join(text.split())


def text_similarity(left: str, right: str) -> float:
    left_norm = normalize_for_compare(left)
    right_norm = normalize_for_compare(right)
    if not left_norm and not right_norm:
        return 1.0
    if not left_norm or not right_norm:
        return 0.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def is_near_duplicate(left: str, right: str, threshold: float) -> bool:
    left_norm = normalize_for_compare(left)
    right_norm = normalize_for_compare(right)
    if not left_norm or not right_norm:
        return False
    if text_similarity(left_norm, right_norm) >= threshold:
        return True
    shorter, longer = sorted([left_norm, right_norm], key=len)
    return len(shorter) >= 6 and shorter in longer and text_similarity(shorter, longer) >= 0.64
