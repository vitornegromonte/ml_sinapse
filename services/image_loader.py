import base64
import io
from pathlib import Path

import requests
from PIL import Image


class ImageLoader:
    def __init__(self, request_timeout: int = 120):
        self._timeout = request_timeout

    def load(self, data: dict) -> Image.Image:
        if "image_url" in data:
            return self._from_url(data["image_url"])

        if "image_base64" in data:
            return self._from_base64(data["image_base64"])

        if "image_path" in data:
            return self._from_path(data["image_path"])

        raise ValueError("Payload must contain one of: image_url, image_base64, image_path")

    def _from_url(self, url: str) -> Image.Image:
        resp = requests.get(url, timeout=self._timeout)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content))

    def _from_base64(self, raw: str) -> Image.Image:
        decoded = base64.b64decode(raw)
        return Image.open(io.BytesIO(decoded))

    def _from_path(self, path: str) -> Image.Image:
        return Image.open(Path(path))
