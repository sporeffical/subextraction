from __future__ import annotations

from pathlib import Path

import requests

from subextraction.config import OCR_PROMPT
from subextraction.ocr.base import OcrResult
from subextraction.preprocessing import image_to_data_url
from subextraction.text import clean_ocr_text


class VllmVisionOcr:
    name = "vllm"

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str | None = None,
        prompt: str = OCR_PROMPT,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.prompt = prompt
        self.timeout_seconds = timeout_seconds

    def resolve_model(self) -> str:
        if self.model:
            return self.model
        response = requests.get(f"{self.base_url}/models", timeout=20)
        response.raise_for_status()
        data = response.json()
        models = data.get("data") or []
        if not models:
            raise RuntimeError("vLLM /models returned no models")
        self.model = models[0]["id"]
        return self.model

    def read(self, crop_path: Path, preprocessed_path: Path | None = None) -> OcrResult:
        try:
            model = self.resolve_model()
            payload = {
                "model": model,
                "temperature": 0,
                "max_tokens": 128,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.prompt},
                            {"type": "image_url", "image_url": {"url": image_to_data_url(crop_path)}},
                        ],
                    }
                ],
            }
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return OcrResult(
                backend=self.name,
                text=clean_ocr_text(content),
                raw=content,
            )
        except Exception as exc:
            return OcrResult(backend=self.name, text="", error=str(exc))
