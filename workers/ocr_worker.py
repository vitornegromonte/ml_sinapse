import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from config import settings
from models.ocr import AbstractOcrModel, OcrTask, create_model
from services.image_loader import ImageLoader
from services.ocr_checkpoint import OcrCheckpoint, checkpoint_key
from services.pdf_extractor import PdfExtractor
from services.result_publisher import ResultPublisher
from services.skew_corrector import SkewCorrector
from services.storage import StorageService
from workers.base_worker import BaseWorker, PermanentTaskError


logger = logging.getLogger(__name__)


class OcrTimeoutError(RuntimeError):
    """A model call exceeded ocr_predict_timeout_seconds.

    The model thread may still be running, so the worker must not reuse the
    model. See OcrWorker._predict.
    """


@dataclass
class _ScannedPage:
    slot: int  # index of this page in the output chunk list
    number: int  # scanned-chunk number, the checkpoint key
    image: Image.Image
    tag: str


class OcrWorker(BaseWorker):
    def __init__(self, cfg=settings):
        super().__init__(cfg.ocr_stream, cfg)
        self._model: AbstractOcrModel | None = None
        self._image_loader = ImageLoader(
            request_timeout=cfg.ocr_request_timeout,
            allowed_root=cfg.ocr_image_allowed_root or None,
        )
        self._skew_corrector = SkewCorrector()
        self._pdf_extractor = PdfExtractor(dpi=cfg.ocr_render_dpi)
        self._result_publisher: ResultPublisher | None = None
        self._storage: StorageService | None = None
        self._model_stuck = False
        self._checkpoint: OcrCheckpoint | None = None

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
        self._checkpoint = None

        # Validate before loading the model: a bad payload must not pay for a
        # model load, and it can never succeed, so it is not retried.
        try:
            task = OcrTask(**data)
        except ValidationError as exc:
            raise PermanentTaskError(f"Invalid OCR task payload: {exc}") from exc
        if not (task.has_pdf() or task.has_image()):
            raise PermanentTaskError(
                "Payload must contain one of: image_url, image_base64, image_path, "
                "or pdf_bucket+pdf_key"
            )

        if self._model is None:
            await asyncio.to_thread(self._load_model)

        if task.has_pdf():
            text = await self._process_pdf(task)
        else:
            text = await self._process_image(task)

        # An empty result is not valid OCR output. Raising sends the task down
        # the retry -> DLQ path instead of publishing nothing for the LLM worker
        # to summarize. One blank page inside a PDF is still allowed.
        if not text.strip():
            raise ValueError(
                f"OCR produced no text for document_id={task.document_id!r}"
            )

        # Never log the OCR text: BOs hold personal data. Length is enough.
        self.logger.info(
            "OCR result: document_id=%s text_length=%d",
            task.document_id,
            len(text),
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

        # Only now is the work final: bronze has the text and the backend was
        # told. A failure above leaves the checkpoint so the retry resumes.
        if self._checkpoint is not None:
            await self._checkpoint.clear()

    async def _process_image(self, task: OcrTask) -> str:
        try:
            image = await asyncio.to_thread(self._image_loader.load, task.model_dump())
        except (ValueError, UnidentifiedImageError) as exc:
            # Bad base64, unknown format, or a path outside the allowed root:
            # the same payload fails every time. Network errors are not caught.
            raise PermanentTaskError(f"Unreadable image: {exc}") from exc
        try:
            # Normalize to RGB first. Skew correction fails on L/P/1 images,
            # and the PNG data URI fails on CMYK. convert() also decodes the
            # pixels, so a truncated file fails here.
            image = await asyncio.to_thread(image.convert, "RGB")
        except OSError as exc:
            raise PermanentTaskError(f"Corrupt image: {exc}") from exc
        image = await asyncio.to_thread(self._skew_corrector.correct, image)
        return (await self._predict([image]))[0]

    async def _predict(self, images: list[Image.Image]) -> list[str]:
        """Run one model call for `images`, bounded by a timeout.

        A Python thread cannot be cancelled: after a timeout the vLLM call
        may still be running on the GPU. Reusing the model would run two calls
        on one engine, which is not thread-safe. So a timeout fails this
        message (the normal retry/DLQ path counts it) and retires the worker:
        it stops reading and run() exits the process for the container
        restart policy to replace.
        """
        try:
            texts = await asyncio.wait_for(
                asyncio.to_thread(
                    self._model.predict_batch, images, self.cfg.ocr_max_tokens
                ),
                timeout=self.cfg.ocr_predict_timeout_seconds,
            )
        except TimeoutError as exc:
            self._model_stuck = True
            self._running = False
            self.logger.critical(
                "Model call exceeded %ds for %d image(s); retiring this worker",
                self.cfg.ocr_predict_timeout_seconds,
                len(images),
            )
            raise OcrTimeoutError(
                f"Model call exceeded {self.cfg.ocr_predict_timeout_seconds}s"
            ) from exc
        if len(texts) != len(images):
            raise RuntimeError(
                f"Model returned {len(texts)} texts for {len(images)} images"
            )
        return texts

    async def _process_pdf(self, task: OcrTask) -> str:
        """Download the PDF, split into text/image chunks, OCR each image chunk.

        One task = one PDF, processed by this one worker (text chunks are
        cheap; scanned pages go to the model in batches, see _ocr_chunks). A
        retry of the same task resumes from the pages already in the Redis
        checkpoint.
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

        cached = await self._open_checkpoint(task, pdf_bytes)

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            pdf_path = Path(tmp.name)

            page_count = await self._pdf_extractor.page_count(pdf_path)
            if page_count > self.cfg.ocr_max_pdf_pages:
                raise PermanentTaskError(
                    f"PDF {task.pdf_bucket}/{task.pdf_key} has {page_count} pages, "
                    f"exceeding ocr_max_pdf_pages={self.cfg.ocr_max_pdf_pages}"
                )

            chunks = await self._ocr_chunks(pdf_path, cached)

        return "\n".join(chunks)

    async def _open_checkpoint(self, task: OcrTask, pdf_bytes: bytes) -> dict[int, str]:
        """Set up the checkpoint for this PDF and return its saved pages."""
        if self.redis is None or self.cfg.ocr_partial_ttl_seconds == 0:
            return {}
        # The text depends on the PDF, the model, the DPI and the token cap.
        settings_tag = (
            f"{self.cfg.ocr_model_name}|{self.cfg.ocr_render_dpi}|{self.cfg.ocr_max_tokens}"
        )
        key = await asyncio.to_thread(
            checkpoint_key, self.stream, task.document_id, pdf_bytes, settings_tag
        )
        self._checkpoint = OcrCheckpoint(self.redis, key, self.cfg.ocr_partial_ttl_seconds)
        cached = await self._checkpoint.load()
        if cached:
            self.logger.info(
                "Resuming document_id=%s: %d scanned page(s) restored from checkpoint",
                task.document_id,
                len(cached),
            )
        return cached

    async def _ocr_chunks(self, pdf_path: Path, cached: dict[int, str]) -> list[str]:
        """Return the PDF's text, OCR-ing scanned pages in batches.

        CPU and GPU work overlap: while one batch is on the model, this loop
        keeps rendering and deskewing the pages of the next batch. At most one
        batch is on the model and one is being prepared, so memory is bounded.
        Pages in `cached` (a retry) are not rendered or OCR-ed again. Each
        finished batch is written to the checkpoint.
        """
        batch_size = self.cfg.ocr_page_batch_size
        chunks: list[str | None] = []
        pending: list[_ScannedPage] = []
        inflight: tuple[asyncio.Task, list[_ScannedPage]] | None = None
        number = 0

        try:
            async for kind, content, tag in self._pdf_extractor.extract_chunks(
                pdf_path, cached_images=cached.keys()
            ):
                if kind == "text":
                    chunks.append(content)
                    continue

                page_number, number = number, number + 1
                if kind == "cached":
                    chunks.append(self._tagged(tag, cached[page_number]))
                    continue

                image = await asyncio.to_thread(self._skew_corrector.correct, content)
                chunks.append(None)
                pending.append(_ScannedPage(len(chunks) - 1, page_number, image, tag))
                if len(pending) >= batch_size:
                    await self._finish_batch(inflight, chunks)
                    inflight = (self._start_batch(pending), pending)
                    pending = []

            await self._finish_batch(inflight, chunks)
            inflight = None
            if pending:
                inflight = (self._start_batch(pending), pending)
                await self._finish_batch(inflight, chunks)
                inflight = None
        finally:
            # An error while preparing pages must not orphan a running model
            # call: wait for it, and discard its result.
            if inflight is not None and not inflight[0].done():
                await asyncio.gather(inflight[0], return_exceptions=True)

        return [text for text in chunks if text is not None]

    def _start_batch(self, pages: list[_ScannedPage]) -> asyncio.Task:
        return asyncio.create_task(self._predict([page.image for page in pages]))

    async def _finish_batch(
        self,
        inflight: tuple[asyncio.Task, list[_ScannedPage]] | None,
        chunks: list[str | None],
    ) -> None:
        """Wait for a started batch, fill its slots, and checkpoint it."""
        if inflight is None:
            return
        task, pages = inflight
        texts = await task
        for page, text in zip(pages, texts):
            chunks[page.slot] = self._tagged(page.tag, text)
        if self._checkpoint is not None:
            await self._checkpoint.save(
                {page.number: text for page, text in zip(pages, texts)}
            )

    @staticmethod
    def _tagged(tag: str, text: str) -> str:
        return f"\n{tag}\n{text}" if tag else text

    def run(self) -> None:
        self.logger.info("Starting OCR worker (pid=%s)", os.getpid())
        try:
            super().run()
        finally:
            if self._model_stuck:
                # Skip unload: the stuck thread still uses the model, and the
                # interpreter would wait for that thread forever at exit. Exit
                # hard so the container restart policy starts a clean process.
                self.logger.critical("OCR worker exiting after a model timeout")
                logging.shutdown()
                os._exit(1)
            else:
                self._unload_model()
                self.logger.info("OCR worker finished")
