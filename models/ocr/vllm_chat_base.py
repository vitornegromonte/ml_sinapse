"""Shared vLLM LLM.chat() mechanics for VLM-based OCR models.

Extracted once a second model (GLM-OCR, migrated from transformers) needed
the exact same load/predict/unload flow as DharmaOcrModel, differing only in
prompt text. Subclasses just set SYSTEM_PROMPT/USER_PROMPT.
"""
import base64
from io import BytesIO
from typing import Any

from PIL import Image

from models.ocr.base import AbstractOcrModel


def _image_to_data_uri(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


class VllmChatOcrModel(AbstractOcrModel):
    """Base for VLM OCR models served via vLLM's LLM.chat() interface.

    GPU selection follows the same convention already used by LLMModel's
    vLLM path (models/llm_model.py): no explicit device pinning
    here, vLLM uses whatever CUDA_VISIBLE_DEVICES the container/deploy
    already restricts it to.
    """

    SYSTEM_PROMPT: str = ""
    USER_PROMPT: str = ""

    def __init__(
        self,
        model_name: str,
        cache_dir: str,
        gpu_device: int,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
    ):
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._gpu_device = gpu_device
        self._dtype = dtype
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._llm: Any = None

    def load(self) -> None:
        from vllm import LLM

        kwargs: dict[str, Any] = dict(
            model=self._model_name,
            trust_remote_code=True,
            dtype=self._dtype,
            gpu_memory_utilization=self._gpu_memory_utilization,
            download_dir=self._cache_dir,
            limit_mm_per_prompt={"image": 1},
        )
        if self._max_model_len is not None:
            kwargs["max_model_len"] = self._max_model_len

        self._llm = LLM(**kwargs)

    def predict(self, image: Image.Image, max_tokens: int, prompt: str | None = None) -> str:
        from vllm import SamplingParams

        if self._llm is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": _image_to_data_uri(image)}},
                {"type": "text", "text": prompt or self.USER_PROMPT},
            ]},
        ]

        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0)
        outputs = self._llm.chat(messages, sampling_params=sampling_params)
        return outputs[0].outputs[0].text.strip()

    def unload(self) -> None:
        self._llm = None
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def device(self) -> str:
        import torch

        return f"cuda:{self._gpu_device}" if torch.cuda.is_available() else "cpu"
