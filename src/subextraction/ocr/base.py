from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from subextraction.text import clean_ocr_text


@dataclass(slots=True)
class OcrResult:
    backend: str
    text: str
    confidence: float | None = None
    error: str | None = None
    raw: str | None = None

    def to_json_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["text"] = clean_ocr_text(self.text)
        return data


class OcrEngine(Protocol):
    name: str

    def read(self, crop_path: Path, preprocessed_path: Path | None = None) -> OcrResult:
        ...
