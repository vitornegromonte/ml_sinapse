import asyncio
import logging

from config import settings
from models.llm_model import LLMModel
from models.tasks import LlmTask
from services.result_publisher import ResultPublisher
from services.storage import StorageService
from workers.base_worker import BaseWorker

logger = logging.getLogger(__name__)


class LlmWorker(BaseWorker):
    """Consome ocr:tasks:results e gera resumos via LLM (vLLM)."""

    supports_batch = True

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

    def _predict_many(self, prompts: list[str]) -> list[str]:
        return [result["text"] for result in self._llm.predict(prompts)]

    async def process(self, data: dict) -> None:
        (error,) = await self.process_batch([data])
        if error is not None:
            raise error

    async def process_batch(self, items: list[dict]) -> list[Exception | None]:
        """One generate() call for the whole batch: vLLM batches internally,
        N prompts in one call instead of N calls of one prompt.

        Payload errors, empty text and bronze/publish failures stay per item,
        so retry/DLQ keeps its per-message meaning.
        """
        errors: list[Exception | None] = [None] * len(items)
        ready: list[tuple[int, LlmTask, dict, str]] = []
        for position, data in enumerate(items):
            try:
                task = LlmTask(**data)
            except Exception as exc:
                errors[position] = exc
                continue
            if not task.text or not task.text.strip():
                self.logger.warning(
                    "Empty text for document_id=%s, skipping", task.document_id
                )
                continue
            ready.append((position, task, data, self._build_prompt(task.text)))
        if not ready:
            return errors

        if self._llm is None:
            await asyncio.to_thread(self._load_model)

        try:
            summaries = await asyncio.to_thread(
                self._predict_many, [item[3] for item in ready]
            )
        except Exception as exc:
            for position, *_ in ready:
                errors[position] = exc
            return errors

        for (position, task, data, _), summary in zip(ready, summaries):
            doc_id = task.document_id
            if not summary.strip():
                self.logger.warning(
                    "Empty summary generated for document_id=%s", doc_id
                )
                continue

            self.logger.info(
                "LLM result: document_id=%s source_length=%d summary_length=%d summary=%.100s",
                doc_id,
                len(task.text),
                len(summary),
                summary,
            )

            result_data = {**data, "text": summary}

            # Bronze before the results notification: save() raises on
            # failure, so a MinIO outage fails the task (retry → DLQ) instead
            # of publishing a result that never landed in bronze.
            try:
                self._ensure_storage()
                self._storage.save(
                    prefix="llm",
                    document_id=doc_id,
                    data=summary.encode("utf-8"),
                )

                self._ensure_publisher()
                if self._result_publisher is not None:
                    await self._result_publisher.publish(result_data, summary)
            except Exception as exc:
                errors[position] = exc

        return errors

    def run(self) -> None:
        try:
            super().run()
        finally:
            self._unload_model()
            self.logger.info("LLM worker finished")
