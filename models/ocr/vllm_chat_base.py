"""Base class for OCR models that run on vLLM through `LLM.chat()`.

Dharma and GLM share the whole load / predict / unload flow. They differ only
in their prompts and a few class settings, which a subclass overrides.

`vllm` and `torch` are imported inside the methods, not at module level. The
test suite runs on CPU without vLLM, and importing this module must not load
them.
"""
import base64
import logging
from io import BytesIO
from typing import Any

from PIL import Image

from models.ocr.base import AbstractOcrModel

logger = logging.getLogger(__name__)


def _image_to_data_uri(image: Image.Image) -> str:
    """Encode an image as a PNG data URI, the form `LLM.chat()` accepts."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


class VllmChatOcrModel(AbstractOcrModel):
    """A vision-language OCR model served by vLLM's `LLM.chat()`.

    GPU choice: this class does not pin a device. vLLM uses the GPU that
    `CUDA_VISIBLE_DEVICES` leaves visible, as the LLM worker does too.

    Subclass settings:
        SYSTEM_PROMPT, USER_PROMPT: the conversation sent with each image.
        MM_LIMITS: how many inputs of each kind one prompt may carry.
        REPETITION_PENALTY: sampling penalty against repeated tokens.
        _postprocess(): turns the raw answer into the page text.
    """

    SYSTEM_PROMPT: str = ""
    USER_PROMPT: str = ""

    # OCR sends one image per prompt and nothing else. A model that also
    # accepts video lists "video": 0. Otherwise vLLM reserves memory to profile
    # a maximum-size video, which OCR never sends.
    MM_LIMITS: dict[str, int] = {"image": 1}

    # Above 1.0, repeated tokens are less likely. A model that loops on a page
    # until max_tokens sets this (see DharmaOcrModel).
    REPETITION_PENALTY: float = 1.0

    def __init__(
        self,
        model_name: str,
        cache_dir: str,
        gpu_device: int,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
    ):
        """
        `gpu_memory_utilization` has vLLM's usual default here. The 15% cap
        from `Settings` applies only when the caller passes the setting, as
        `OcrWorker` does.
        """
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._gpu_device = gpu_device
        self._dtype = dtype
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._llm: Any = None

    # public

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

    def predict(self, image: Image.Image, max_tokens: int, prompt: str | None = None) -> str:
        self._require_loaded()

        messages = self._build_messages(image, prompt)
        outputs = self._llm.chat(messages, sampling_params=self._sampling_params(max_tokens))
        return self._completion_text(outputs[0], max_tokens)

    def predict_batch(
        self, images: list[Image.Image], max_tokens: int, prompt: str | None = None
    ) -> list[str]:
        """One `LLM.chat()` call for all images; vLLM batches them on the GPU."""
        self._require_loaded()
        if not images:
            return []

        conversations = [self._build_messages(image, prompt) for image in images]
        outputs = self._llm.chat(
            conversations, sampling_params=self._sampling_params(max_tokens)
        )
        if len(outputs) != len(images):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(images)} images"
            )
        return [self._completion_text(output, max_tokens) for output in outputs]

    def unload(self) -> None:
        self._llm = None  # first, so the engine is released even if an import fails
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def device(self) -> str:
        import torch

        return f"cuda:{self._gpu_device}" if torch.cuda.is_available() else "cpu"

    # subclass hook

    def _postprocess(self, text: str) -> str:
        """Turn the model's raw answer into the page text.

        The default returns the answer as it is. A model that wraps its answer,
        for example in JSON, overrides this.
        """
        return text

    # private

    def _require_loaded(self) -> None:
        if self._llm is None:
            raise RuntimeError("Model not loaded. Call load() first.")

    def _build_messages(self, image: Image.Image, prompt: str | None) -> list[dict]:
        """One conversation: the system prompt, then the image and user prompt."""
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": _image_to_data_uri(image)}},
                {"type": "text", "text": prompt or self.USER_PROMPT},
            ]},
        ]

    def _sampling_params(self, max_tokens: int) -> Any:
        """Greedy decoding. The repetition penalty is sent only when it is set."""
        from vllm import SamplingParams

        kwargs: dict[str, Any] = {"max_tokens": max_tokens, "temperature": 0}
        if self.REPETITION_PENALTY != 1.0:
            kwargs["repetition_penalty"] = self.REPETITION_PENALTY
        return SamplingParams(**kwargs)

    def _completion_text(self, output: Any, max_tokens: int) -> str:
        """Read the page text from one vLLM output, and warn if it was cut off."""
        completion = output.outputs[0]
        if completion.finish_reason == "length":
            logger.warning(
                "OCR output hit max_tokens=%d and was truncated; "
                "raise ocr_max_tokens or lower ocr_render_dpi",
                max_tokens,
            )
        return self._postprocess(completion.text.strip())
