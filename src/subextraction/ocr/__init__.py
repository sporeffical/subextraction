from __future__ import annotations

from subextraction.config import ExtractionConfig
from subextraction.ocr.base import OcrEngine
from subextraction.ocr.tesseract import TesseractOcr
from subextraction.ocr.vllm import VllmVisionOcr


def build_ocr_engines(config: ExtractionConfig) -> list[OcrEngine]:
    if config.ocr_backend == "vllm":
        return [
            VllmVisionOcr(
                base_url=config.vllm_base_url,
                model=config.vllm_model,
                prompt=config.prompt,
                timeout_seconds=config.vllm_timeout_seconds,
            )
        ]
    if config.ocr_backend == "tesseract":
        return [TesseractOcr(tesseract_cmd=config.tesseract_cmd)]
    return [
        VllmVisionOcr(
            base_url=config.vllm_base_url,
            model=config.vllm_model,
            prompt=config.prompt,
            timeout_seconds=config.vllm_timeout_seconds,
        ),
        TesseractOcr(tesseract_cmd=config.tesseract_cmd),
    ]
