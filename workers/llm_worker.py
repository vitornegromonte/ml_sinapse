import asyncio
import logging

from config import settings
from models.llm_model import LLMModel
from services.result_publisher import ResultPublisher
from services.storage import StorageService
from workers.base_worker import BaseWorker

logger = logging.getLogger(__name__)


class LlmWorker(BaseWorker):
    """Consome ocr:tasks:results e gera resumos via LLM (vLLM)."""

    def __init__(self, cfg=settings):
        super().__init__(cfg.ocr_stream + ":results", cfg)
        self._llm: LLMModel | None = None
        self._result_publisher: ResultPublisher | None = None
        self._storage: StorageService | None = None

    def _load_model(self) -> None:
        self._llm = LLMModel(
            model_path=self.cfg.llm_model_path,
            device=f"cuda:{self.cfg.llm_gpu_device}",
            gpu_memory_utilization=self.cfg.llm_gpu_memory_utilization,
            max_model_len=self.cfg.llm_max_model_len,
            tensor_parallel_size=self.cfg.llm_tensor_parallel_size,
            dtype=self.cfg.llm_dtype,
            sampling_params=self.cfg.llm_sampling_params or {},
        )
        self._llm.load()
        self.logger.info("LLM model loaded (vLLM, %s)", self.cfg.llm_model_path)

    def _unload_model(self) -> None:
        if self._llm is not None:
            self.logger.info("Unloading LLM model")
            self._llm.unload()
            self._llm = None

    def _ensure_storage(self) -> None:
        if self._storage is None:
            self._storage = StorageService(
                endpoint=self.cfg.minio_endpoint,
                access_key=self.cfg.minio_access_key,
                secret_key=self.cfg.minio_secret_key,
                bucket=self.cfg.minio_bucket_bronze,
            )

    def _ensure_publisher(self) -> None:
        if self._result_publisher is None and self.redis is not None:
            self._result_publisher = ResultPublisher(
                redis=self.redis,
                stream=self.cfg.llm_stream,
                maxlen=self.cfg.llm_result_stream_maxlen,
            )

    def _build_prompt(self, text: str) -> str:
        return self.cfg.llm_prompt_template.format(text=text)

    def _predict(self, prompt: str) -> str:
        results = self._llm.predict(prompt)
        return results[0]["text"]

    async def process(self, data: dict) -> None:
        if self._llm is None:
            await asyncio.to_thread(self._load_model)

        doc_id = data.get("document_id")
        if not doc_id:
            raise ValueError("Payload must contain 'document_id'")

        text = data.get("text")
        if not text or not text.strip():
            self.logger.warning(
                "Empty text for document_id=%s, skipping", doc_id
            )
            return

        prompt = self._build_prompt(text)
        summary = await asyncio.to_thread(self._predict, prompt)

        if not summary.strip():
            self.logger.warning(
                "Empty summary generated for document_id=%s", doc_id
            )
            return

        self.logger.info(
            "LLM result: document_id=%s source_length=%d summary_length=%d summary=%.100s",
            doc_id,
            len(text),
            len(summary),
            summary,
        )

        result_data = {**data, "text": summary}

        self._ensure_publisher()
        if self._result_publisher is not None:
            await self._result_publisher.publish(result_data, summary)

        self._ensure_storage()
        if self._storage is not None:
            self._storage.save(
                prefix="llm",
                document_id=doc_id,
                data=summary.encode("utf-8"),
            )

    def run(self) -> None:
        try:
            super().run()
        finally:
            self._unload_model()
            self.logger.info("LLM worker finished")
