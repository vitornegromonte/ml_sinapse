"""Shared vLLM LLM.chat() mechanics for VLM-based OCR models.

Extracted once a second model (GLM-OCR, migrated from transformers) needed
the exact same load/predict/unload flow as DharmaOcrModel, differing only in
prompt text. Subclasses just set SYSTEM_PROMPT/USER_PROMPT.
"""
import base64
import logging
from io import BytesIO
from typing import Any

from PIL import Image

from models.ocr.base import AbstractOcrModel

logger = logging.getLogger(__name__)


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

    # Inputs per prompt, by modality. OCR sends one image and nothing else.
    # A model that also supports video lists "video": 0, or vLLM profiles
    # memory for a maximum-size video and reserves GBs that OCR never uses.
    MM_LIMITS: dict[str, int] = {"image": 1}
    # Values above 1.0 make repeating tokens less likely. A model that tends to
    # loop on a page until max_tokens sets this (see DharmaOcrModel).
    REPETITION_PENALTY: float = 1.0
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
            limit_mm_per_prompt=dict(self.MM_LIMITS),
        )
        if self._max_model_len is not None:
            kwargs["max_model_len"] = self._max_model_len

        self._llm = LLM(**kwargs)

    def _build_messages(self, image: Image.Image, prompt: str | None) -> list[dict]:
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": _image_to_data_uri(image)}},
                {"type": "text", "text": prompt or self.USER_PROMPT},
            ]},
        ]

    def _sampling_params(self, max_tokens: int) -> Any:
        from vllm import SamplingParams

        kwargs: dict[str, Any] = {"max_tokens": max_tokens, "temperature": 0}
        if self.REPETITION_PENALTY != 1.0:
            kwargs["repetition_penalty"] = self.REPETITION_PENALTY
        return SamplingParams(**kwargs)

    def _postprocess(self, text: str) -> str:
        """Turn the model's raw output into the page text. Identity by default;
        a model that wraps its output (for example in JSON) overrides this."""
        return text

    def _completion_text(self, output: Any, max_tokens: int) -> str:
        completion = output.outputs[0]
        # finish_reason "length" means the model hit max_tokens: the page text
        # is cut off. Warn so the operator can raise OCR_MAX_TOKENS.
        if completion.finish_reason == "length":
            logger.warning(
                "OCR output hit max_tokens=%d and was truncated; "
                "raise ocr_max_tokens or lower ocr_render_dpi",
                max_tokens,
            )
        return self._postprocess(completion.text.strip())

    def predict(self, image: Image.Image, max_tokens: int, prompt: str | None = None) -> str:
        if self._llm is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        messages = self._build_messages(image, prompt)
        sampling_params = self._sampling_params(max_tokens)
        outputs = self._llm.chat(messages, sampling_params=sampling_params)
        return self._completion_text(outputs[0], max_tokens)

    def predict_batch(
        self, images: list[Image.Image], max_tokens: int, prompt: str | None = None
    ) -> list[str]:
        """One LLM.chat() call for all images; vLLM batches them on the GPU."""
        if self._llm is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        if not images:
            return []

        conversations = [self._build_messages(image, prompt) for image in images]
        sampling_params = self._sampling_params(max_tokens)
        outputs = self._llm.chat(conversations, sampling_params=sampling_params)
        if len(outputs) != len(images):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(images)} images"
            )
        return [self._completion_text(output, max_tokens) for output in outputs]

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
