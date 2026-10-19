import base64
import io
from pathlib import Path

import requests
from PIL import Image


class ImageLoader:
    """Loads an image from a URL, a base64 blob, or a local path.

    A local path is only read from inside `allowed_root`: the path is resolved
    (symlinks followed) and must stay under that directory. `image_path` comes
    from the task payload, so without a confinement root it is an
    arbitrary-file-read vector — no `allowed_root` means local paths are
    rejected outright.
    """

    def __init__(
        self,
        request_timeout: int = 120,
        allowed_root: str | Path | None = None,
    ):
        self._timeout = request_timeout
        self._allowed_root = Path(allowed_root).resolve() if allowed_root else None

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
        return Image.open(self._resolve_local_path(path))

    def _resolve_local_path(self, path: str) -> Path:
        if self._allowed_root is None:
            raise ValueError(
                "image_path is disabled: set ocr_image_allowed_root "
                "(OCR_IMAGE_ALLOWED_ROOT) to the directory local images may be "
                "read from"
            )
        raw = Path(path)
        candidate = (raw if raw.is_absolute() else self._allowed_root / raw).resolve()
        if not candidate.is_relative_to(self._allowed_root):
            raise ValueError(
                f"image_path {path!r} escapes the allowed root {self._allowed_root}"
            )
        return candidate
