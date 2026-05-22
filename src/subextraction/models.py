from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class VideoInfo:
    path: Path
    frame_count: int
    fps: float
    width: int
    height: int
    duration_seconds: float

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        return data


@dataclass(slots=True)
class FrameSample:
    frame_index: int
    timestamp_seconds: float
    image: Any


@dataclass(slots=True)
class CropDraft:
    frame_index: int
    timestamp_seconds: float
    frame_image: Any
    crop_image: Any
    preprocessed_image: Any
    bbox: tuple[int, int, int, int]
    signal_score: float
    diff_score: float


@dataclass(slots=True)
class CropArtifact:
    frame_index: int
    timestamp_seconds: float
    frame_path: Path
    crop_path: Path
    preprocessed_path: Path
    bbox: tuple[int, int, int, int]
    signal_score: float
    diff_score: float

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "frame_path": str(self.frame_path),
            "crop_path": str(self.crop_path),
            "preprocessed_path": str(self.preprocessed_path),
            "bbox": list(self.bbox),
            "signal_score": self.signal_score,
            "diff_score": self.diff_score,
        }


@dataclass(slots=True)
class OcrObservation:
    frame_index: int
    timestamp_seconds: float
    crop_path: Path
    preprocessed_path: Path
    selected_text: str
    results: list[dict[str, Any]]
    flags: list[str]
    signal_score: float
    diff_score: float

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "crop_path": str(self.crop_path),
            "preprocessed_path": str(self.preprocessed_path),
            "selected_text": self.selected_text,
            "results": self.results,
            "flags": self.flags,
            "signal_score": self.signal_score,
            "diff_score": self.diff_score,
        }


@dataclass(slots=True)
class SubtitleSegment:
    start_seconds: float
    end_seconds: float
    text: str
    frame_indices: list[int]
    observation_count: int

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)
