from __future__ import annotations

import logging
import os
from typing import Any

from models.base_model import BaseModelWrapper

logger = logging.getLogger(__name__)


class LLMModel(BaseModelWrapper):
    """vLLM-backed causal LM wrapper. vLLM is the only supported engine.

    There used to be transformers/llama.cpp/bentoml/mlflow branches, none of
    which were reachable (the worker always passed engine="vllm") and two of
    which imported packages that are not dependencies. Deleted rather than
    kept behind an adapter — see docs/adr-010-bentoml-serving.md for the
    BentoML approach that was dropped.

    Model-specific behavior comes from the caller, not from this class:
      sampling_params: vLLM SamplingParams fields. Always set "max_tokens":
          vLLM's own default is 16, which cuts a summary off after one line.
      chat_template_kwargs: variables for the model's chat template, for
          example {"enable_thinking": False} to turn off a reasoning mode.
          A template that does not use a variable ignores it.
      engine_kwargs: extra arguments for the vLLM engine, for example
          {"language_model_only": True} to skip a vision encoder.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        tensor_parallel_size: int = 1,
        sampling_params: dict | None = None,
        chat_template_kwargs: dict | None = None,
        engine_kwargs: dict | None = None,
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.sampling_params = sampling_params or {}
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.engine_kwargs = engine_kwargs or {}

        if "max_tokens" not in self.sampling_params:
            logger.warning(
                "sampling_params has no max_tokens: vLLM will stop after 16 tokens "
                "and every summary will be cut off"
            )

        self._llm: Any = None
        self._tokenizer: Any = None

    def load(self) -> None:
        from vllm import LLM as VLLM
        from transformers import AutoTokenizer

        # vLLM pins a specific GPU index via CUDA_VISIBLE_DEVICES, not a
        # constructor kwarg — same convention as models/ocr/dharma.py.
        if self.device.startswith("cuda:"):
            os.environ["CUDA_VISIBLE_DEVICES"] = self.device.split(":", 1)[1]

        kwargs: dict[str, Any] = {
            "model": self.model_path,
            "dtype": self.dtype,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "tensor_parallel_size": self.tensor_parallel_size,
        }
        if self.max_model_len is not None:
            kwargs["max_model_len"] = self.max_model_len
        kwargs.update(self.engine_kwargs)

        self._llm = VLLM(**kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        logger.info(
            "vLLM engine loaded (model=%s, gpu_mem=%.2f, tp=%d)",
            self.model_path,
            self.gpu_memory_utilization,
            self.tensor_parallel_size,
        )

    def predict(self, inputs: Any, **kwargs: Any) -> Any:
        if self._llm is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        from vllm import SamplingParams

        sampling_params = SamplingParams(**{**self.sampling_params, **kwargs})
        prompts = [inputs] if isinstance(inputs, str) else inputs

        if self._tokenizer is not None:
            prompts = [
                self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                    **self.chat_template_kwargs,
                )
                for p in prompts
            ]

        outputs = self._llm.generate(prompts, sampling_params)
        return [
            {"text": o.outputs[0].text, "tokens": o.outputs[0].token_ids}
            for o in outputs
        ]

    def unload(self) -> None:
        self._llm = None
        self._tokenizer = None
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("Model unloaded")
