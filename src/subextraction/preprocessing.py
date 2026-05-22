from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """Create a high-contrast crop for scoring and literal OCR fallback."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    scaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.fastNlMeansDenoising(scaled, None, h=10, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)
    return cv2.adaptiveThreshold(
        contrast,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )


def image_fingerprint(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, (96, 24), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32) / 255.0


def fingerprint_diff(left: np.ndarray | None, right: np.ndarray) -> float:
    if left is None:
        return 1.0
    return float(np.mean(np.abs(left - right)))


def image_to_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    media_type = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"
