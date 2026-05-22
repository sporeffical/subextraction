from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping

from .models import OcrObservation


def ensure_run_dirs(run_dir: Path) -> dict[str, Path]:
    dirs = {
        "run": run_dir,
        "frames": run_dir / "frames",
        "crops": run_dir / "crops",
        "preprocessed": run_dir / "preprocessed",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_ocr_observations(path: Path) -> list[OcrObservation]:
    observations: list[OcrObservation] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            observations.append(
                OcrObservation(
                    frame_index=int(row["frame_index"]),
                    timestamp_seconds=float(row["timestamp_seconds"]),
                    crop_path=Path(str(row.get("crop_path", ""))),
                    preprocessed_path=Path(str(row.get("preprocessed_path", ""))),
                    selected_text=str(row.get("selected_text", "")),
                    results=list(row.get("results") or []),
                    flags=list(row.get("flags") or []),
                    signal_score=float(row.get("signal_score", 0.0)),
                    diff_score=float(row.get("diff_score", 0.0)),
                )
            )
    return observations


def write_benchmark_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    fieldnames = [
        "frame_index",
        "timestamp_seconds",
        "crop_path",
        "selected_text",
        "vllm_text",
        "tesseract_text",
        "expected_text",
        "failure_notes",
        "flags",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
