from __future__ import annotations

from pathlib import Path

from subextraction.ocr.base import OcrResult
from subextraction.text import clean_ocr_text


class TesseractOcr:
    name = "tesseract"

    def __init__(self, tesseract_cmd: str | None = None) -> None:
        self.tesseract_cmd = tesseract_cmd

    def read(self, crop_path: Path, preprocessed_path: Path | None = None) -> OcrResult:
        try:
            import pytesseract

            if self.tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

            image_path = preprocessed_path or crop_path
            data = pytesseract.image_to_data(
                str(image_path),
                config="--psm 6",
                output_type=pytesseract.Output.DICT,
            )
            words: list[str] = []
            confidences: list[float] = []
            for word, confidence in zip(data.get("text", []), data.get("conf", [])):
                word = str(word).strip()
                if not word:
                    continue
                try:
                    conf = float(confidence)
                except ValueError:
                    conf = -1.0
                if conf >= 0:
                    confidences.append(conf)
                words.append(word)
            text = clean_ocr_text(" ".join(words))
            avg_conf = sum(confidences) / len(confidences) if confidences else None
            return OcrResult(backend=self.name, text=text, confidence=avg_conf, raw=str(data))
        except Exception as exc:
            return OcrResult(backend=self.name, text="", error=str(exc))
