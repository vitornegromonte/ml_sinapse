import asyncio
import logging
import os
import tempfile
from pathlib import Path

from config import settings
from models.ocr import AbstractOcrModel, OcrTask, create_model
from services.image_loader import ImageLoader
from services.pdf_extractor import PdfExtractor
from services.result_publisher import ResultPublisher
from services.skew_corrector import SkewCorrector
from services.storage import StorageService
from workers.base_worker import BaseWorker


logger = logging.getLogger(__name__)


class OcrWorker(BaseWorker):
    def __init__(self, cfg=settings):
        super().__init__(cfg.ocr_stream, cfg)
        self._model: AbstractOcrModel | None = None
        self._image_loader = ImageLoader(
            request_timeout=cfg.ocr_request_timeout,
            allowed_root=cfg.ocr_image_allowed_root or None,
        )
        self._skew_corrector = SkewCorrector()
        self._pdf_extractor = PdfExtractor()
        self._result_publisher: ResultPublisher | None = None
        self._storage: StorageService | None = None

    def _load_model(self) -> None:
        self._model = create_model(
            model_name=self.cfg.ocr_model_name,
            cache_dir=self.cfg.model_base_path,
            gpu_device=self.cfg.ocr_gpu_device,
            model_uri=self.cfg.ocr_model_uri,
            dtype=self.cfg.ocr_dtype,
            gpu_memory_utilization=self.cfg.ocr_gpu_memory_utilization,
            max_model_len=self.cfg.ocr_max_model_len,
        )
        self._model.load()
        self.logger.info(
            "Model loaded on %s: %s",
            self._model.device,
            type(self._model).__name__,
        )

    def _unload_model(self) -> None:
        if self._model is not None:
            self.logger.info("Unloading model")
            self._model.unload()
            self._model = None

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
                stream=self.stream,
                maxlen=self.cfg.ocr_result_stream_maxlen,
            )

    async def process(self, data: dict) -> None:
        if self._model is None:
            await asyncio.to_thread(self._load_model)

        task = OcrTask(**data)
        if task.has_pdf():
            text = await self._process_pdf(task)
        elif task.has_image():
            text = await self._process_image(task)
        else:
            raise ValueError(
                "Payload must contain one of: image_url, image_base64, image_path, "
                "or pdf_bucket+pdf_key"
            )

        self.logger.info(
            "OCR result: document_id=%s text_length=%d text=%.100s",
            task.document_id,
            len(text),
            text,
        )

        # Bronze before the results notification: save() raises on failure, so
        # a MinIO outage fails the task (retry → DLQ) instead of publishing a
        # result for text that never landed in bronze.
        self._ensure_storage()
        self._storage.save(
            prefix="ocr",
            document_id=task.document_id,
            data=text.encode("utf-8"),
        )

        self._ensure_publisher()
        if self._result_publisher is not None:
            await self._result_publisher.publish(task.model_dump(), text)

    async def _process_image(self, task: OcrTask) -> str:
        image = await asyncio.to_thread(self._image_loader.load, task.model_dump())
        image = await asyncio.to_thread(self._skew_corrector.correct, image)
        return await asyncio.to_thread(self._model.predict, image, self.cfg.ocr_max_tokens)

    async def _process_pdf(self, task: OcrTask) -> str:
        """Download the PDF, split into text/image chunks, OCR each image chunk.

        One task = one PDF, processed serially by this one worker (text
        chunks are cheap; image chunks each cost a model.predict() call).
        Horizontal scale comes from other worker replicas picking up other
        *documents* concurrently via the consumer group — not from splitting
        one document's pages across workers. ocr_max_pdf_pages guards against
        a pathological document hanging this worker indefinitely: exceeding
        it raises, which feeds the existing retry/DLQ path in base_worker.py.
        """
        self._ensure_storage()
        pdf_bytes = await asyncio.to_thread(
            self._storage.download, task.pdf_bucket, task.pdf_key
        )

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            pdf_path = Path(tmp.name)

            page_count = await self._pdf_extractor.page_count(pdf_path)
            if page_count > self.cfg.ocr_max_pdf_pages:
                raise ValueError(
                    f"PDF {task.pdf_bucket}/{task.pdf_key} has {page_count} pages, "
                    f"exceeding ocr_max_pdf_pages={self.cfg.ocr_max_pdf_pages}"
                )

            chunks: list[str] = []
            async for kind, content, tag in self._pdf_extractor.extract_chunks(pdf_path):
                if kind == "text":
                    chunks.append(content)
                    continue

                image = await asyncio.to_thread(self._skew_corrector.correct, content)
                page_text = await asyncio.to_thread(
                    self._model.predict, image, self.cfg.ocr_max_tokens
                )
                chunks.append(f"\n{tag}\n{page_text}" if tag else page_text)

        return "\n".join(chunks)

    def run(self) -> None:
        self.logger.info("Starting OCR worker (pid=%s)", os.getpid())
        try:
            super().run()
        finally:
            self._unload_model()
            self.logger.info("OCR worker finished")
