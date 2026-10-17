from __future__ import annotations

import logging
from typing import Any

import bentoml

from config import settings

logger = logging.getLogger(__name__)


@bentoml.service(
    name="sinapse-llm",
    traffic={"timeout": 300},
    resources={
        "gpu": 1,
        "gpu_memory_utilization": settings.llm_gpu_memory_utilization,
    },
)
class LLMBentoService:
    def __init__(self) -> None:
        from models.llm_model import LLMModel

        self.model = LLMModel(
            model_path=settings.llm_model_path,
            engine=settings.llm_engine,
            dtype=settings.llm_dtype,
            gpu_memory_utilization=settings.llm_gpu_memory_utilization,
            max_model_len=settings.llm_max_model_len,
            tensor_parallel_size=settings.llm_tensor_parallel_size,
            sampling_params=settings.llm_sampling_params or {},
        )
        self.model.load()
        logger.info(
            "BentoML service initialized (model=%s, engine=%s)",
            settings.llm_model_path,
            settings.llm_engine,
        )

    @bentoml.api(batchable=False)
    def generate(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        result = self.model.predict(prompt, **kwargs)
        return result[0]

    @bentoml.api(batchable=True, max_batch_size=32)
    def generate_batch(
        self, prompts: list[str], **kwargs: Any
    ) -> list[dict[str, Any]]:
        return self.model.predict(prompts, **kwargs)
