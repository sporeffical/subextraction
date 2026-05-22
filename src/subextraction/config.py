from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal


OCR_PROMPT = """Read only overlaid subtitle/intertitle text visible in this image.
Return an empty string if there is no subtitle or intertitle.
Do not describe, interpret, summarize, or identify the scene.
Do not transcribe objects, clothing, signs, logos, years, titles, or labels unless they are clearly part of the overlaid subtitle/intertitle.
Do not guess missing words.
Preserve punctuation and line breaks when clear."""


OcrBackend = Literal["vllm", "tesseract", "both"]
CropMode = Literal["bright", "monochrome"]


@dataclass(slots=True)
class ExtractionConfig:
    video_path: Path
    output_root: Path = Path("output")
    run_name: str | None = None
    scan_stride: int = 12
    bottom_fraction: float = 0.34
    crop_mode: CropMode = "bright"
    crop_padding_x: int = 24
    crop_padding_y: int = 18
    min_signal: float = 0.0015
    min_diff: float = 0.030
    force_every_seconds: float = 1.8
    start_seconds: float | None = None
    end_seconds: float | None = None
    limit: int | None = None
    ocr_backend: OcrBackend = "vllm"
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str | None = None
    vllm_timeout_seconds: float = 120.0
    tesseract_cmd: str | None = None
    merge_similarity: float = 0.86
    merge_gap_seconds: float = 4.0
    min_segment_seconds: float = 0.65
    hold_seconds: float = 1.2
    prompt: str = OCR_PROMPT

    def make_run_dir(self, mode: str) -> Path:
        if self.run_name:
            name = self.run_name
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"{self.video_path.stem}_{mode}_{stamp}"
        return self.output_root / name

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["video_path"] = str(self.video_path)
        data["output_root"] = str(self.output_root)
        return data
