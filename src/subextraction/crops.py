from __future__ import annotations

import cv2
import numpy as np

CropMode = str


def crop_subtitle_region(
    frame: np.ndarray,
    bottom_fraction: float = 0.34,
    min_mask_ratio: float = 0.0008,
    mode: CropMode = "bright",
    padding_x: int = 24,
    padding_y: int = 18,
) -> tuple[np.ndarray, tuple[int, int, int, int], float]:
    """Crop the likely subtitle area from a frame.

    The method first limits attention to the lower portion of the frame, then
    uses white/yellow HSV masks plus dilation to tighten the crop around text.
    If no useful text-like signal is found, it returns the lower region.
    """
    height, width = frame.shape[:2]
    fraction = min(max(bottom_fraction, 0.10), 0.60)
    roi_y0 = int(height * (1.0 - fraction))
    roi = frame[roi_y0:height, 0:width]

    mask = _subtitle_mask(roi, mode)
    signal_score = float(np.count_nonzero(mask) / mask.size)
    if signal_score < min_mask_ratio:
        return roi.copy(), (0, roi_y0, width, height), signal_score

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < 40:
            continue
        if w < width * 0.015 or h < 4:
            continue
        boxes.append((x, y, x + w, y + h))

    if not boxes:
        return roi.copy(), (0, roi_y0, width, height), signal_score

    pad_x = max(0, int(padding_x))
    pad_y = max(0, int(padding_y))
    x0 = max(0, min(box[0] for box in boxes) - pad_x)
    y0 = max(0, min(box[1] for box in boxes) - pad_y)
    x1 = min(width, max(box[2] for box in boxes) + pad_x)
    y1 = min(roi.shape[0], max(box[3] for box in boxes) + pad_y)

    min_height = max(48, int(height * 0.08))
    if y1 - y0 < min_height:
        center = (y0 + y1) // 2
        y0 = max(0, center - min_height // 2)
        y1 = min(roi.shape[0], y0 + min_height)

    crop = roi[y0:y1, x0:x1].copy()
    return crop, (x0, roi_y0 + y0, x1, roi_y0 + y1), signal_score


def _subtitle_mask(roi: np.ndarray, mode: CropMode) -> np.ndarray:
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    value_floor = 160 if mode == "monochrome" else 175
    saturation_ceiling = 95 if mode == "monochrome" else 85
    white = cv2.inRange(
        hsv,
        np.array([0, 0, value_floor]),
        np.array([180, saturation_ceiling, 255]),
    )
    yellow = cv2.inRange(hsv, np.array([14, 55, 110]), np.array([45, 255, 255]))
    return cv2.bitwise_or(white, yellow)
