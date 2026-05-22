from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2

from .models import FrameSample, VideoInfo


def get_video_info(video_path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0.0
        return VideoInfo(
            path=video_path,
            frame_count=frame_count,
            fps=fps,
            width=width,
            height=height,
            duration_seconds=duration,
        )
    finally:
        cap.release()


def iter_sampled_frames(
    video_path: Path,
    stride: int,
    fps: float,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> Iterator[FrameSample]:
    if stride < 1:
        raise ValueError("scan_stride must be at least 1")

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = int((start_seconds or 0.0) * fps)
        end_frame = int(end_seconds * fps) if end_seconds is not None else frame_count - 1
        end_frame = min(end_frame, frame_count - 1)

        current_frame = max(0, start_frame)
        end_frame = max(current_frame, end_frame)
        next_sample = current_frame

        if current_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        while current_frame <= end_frame:
            if current_frame == next_sample:
                ok, frame = cap.read()
                if not ok:
                    break
                yield FrameSample(
                    frame_index=current_frame,
                    timestamp_seconds=current_frame / fps if fps > 0 else 0.0,
                    image=frame,
                )
                current_frame += 1
                next_sample += stride
                continue

            ok = cap.grab()
            if not ok:
                break
            current_frame += 1
    finally:
        cap.release()
