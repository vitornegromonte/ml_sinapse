"""The OCR worker: turns an image or a PDF into text.

It consumes `ocr:tasks`. For each task it does these steps:

  1. Validate the payload. A bad payload is a permanent failure (ADR-011).
  2. Load the OCR model. This happens once, on the first task.
  3. Extract the text.
       Image task: load the image, convert it to RGB, straighten it, OCR it.
       PDF task: download the PDF. Keep the text layer as it is. Straighten
       and OCR the scanned parts, several pages per model call.
  4. Reject an empty result.
  5. Save the text to MinIO (bronze), then publish it to `ocr:tasks:results`.
  6. Delete the checkpoint of the PDF's finished pages.

Reliability, in short (see docs/adr-011 and docs/adr-012):
  - A PDF's finished pages are checkpointed in Redis after each batch. A retry
    of the same task resumes from them.
  - While one batch is on the GPU, the next batch is rendered and straightened.
  - A model call that exceeds its time limit retires the worker process.
"""
import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from config import Settings, settings
from models.ocr import AbstractOcrModel, OcrTask, create_model
from services.image_loader import ImageLoader
from services.model_manifest import build_manifest, publish_manifest, revision_for
from services.ocr_checkpoint import OcrCheckpoint, checkpoint_key
from services.pdf_extractor import PdfExtractor
from services.result_publisher import ResultPublisher
from services.skew_corrector import SkewCorrector
from services.storage import StorageService
from workers.base_worker import BaseWorker, PermanentTaskError


class OcrTimeoutError(RuntimeError):
    """A model call exceeded `ocr_predict_timeout_seconds`.

    The call may still be running on the GPU, so the worker must not reuse the
    model. See OcrWorker._predict.
    """


@dataclass
class _ScannedPage:
    """A scanned page, ready for the model."""

    slot: int  # position of the page's text in the output chunk list
    number: int  # scanned-chunk number: the key of the page in the checkpoint
    image: Image.Image
    tag: str  # marker put before the page's text


@dataclass
class _Batch:
    """Pages sent to the model together, and the task that runs the call."""

    task: "asyncio.Task[list[str]]"
    pages: list[_ScannedPage]


class OcrWorker(BaseWorker):
    """Consumes `ocr:tasks` and publishes OCR text. See the module docstring."""

    def __init__(self, cfg: Settings = settings):
        super().__init__(cfg.ocr_stream, cfg)

        # Helpers that need no network at start-up.
        self._image_loader = ImageLoader(
            request_timeout=cfg.ocr_request_timeout,
            allowed_root=cfg.ocr_image_allowed_root or None,
        )
        self._skew_corrector = SkewCorrector()
        self._pdf_extractor = PdfExtractor(dpi=cfg.ocr_render_dpi)

        # What exactly this worker runs (model, pin, prompts, settings). Its id
        # goes on every output; see services/model_manifest.py and ADR-014.
        self._manifest = build_manifest("ocr", cfg)
        self._manifest_published = False

        # Created on first use.
        self._model: AbstractOcrModel | None = None
        self._storage: StorageService | None = None
        self._result_publisher: ResultPublisher | None = None

        # State of the task in progress. The worker handles one task at a time.
        self._checkpoint: OcrCheckpoint | None = None

        # Set when a model call timed out. See _predict and run.
        self._model_stuck = False

    # entry points

    async def process(self, data: dict) -> None:
        self._checkpoint = None
        task = self._parse_task(data)
        await self._ensure_model()

        if task.has_pdf():
            text = await self._process_pdf(task)
        else:
            text = await self._process_image(task)

        # An empty result is not valid OCR output. Raising sends the task down
        # the retry -> DLQ path, instead of publishing nothing for the LLM
        # worker to summarize. One blank page inside a PDF is still allowed.
        if not text.strip():
            raise ValueError(f"OCR produced no text for document_id={task.document_id!r}")

        # Never log the text: BOs hold personal data. The length is enough.
        self.logger.info(
            "OCR result: document_id=%s text_length=%d", task.document_id, len(text)
        )

        # Bronze first, then the notification. save() raises on failure, so a
        # MinIO outage fails the task (retry -> DLQ) instead of announcing text
        # that never reached bronze.
        self._save_to_bronze(task, text)
        await self._publish_result(task, text)

        # Only now is the work final. A failure above keeps the checkpoint, so
        # the retry resumes.
        await self._clear_checkpoint()

    def run(self) -> None:
        self.logger.info("Starting OCR worker (pid=%s)", os.getpid())
        try:
            super().run()
        finally:
            if self._model_stuck:
                # A model call timed out and its thread may still use the model.
                # Do not unload it: the thread would crash, and at exit the
                # interpreter would wait for the thread forever. Exit at once,
                # so the container restart policy starts a clean process.
                self.logger.critical("OCR worker exiting after a model timeout")
                logging.shutdown()
                os._exit(1)
            else:
                self._unload_model()
                self.logger.info("OCR worker finished")

    # task input

    @staticmethod
    def _parse_task(data: dict) -> OcrTask:
        """Validate the payload. Raise PermanentTaskError if it can never work.

        This runs before the model loads: a bad payload must not pay for a
        model load, and retrying it is pointless.
        """
        try:
            task = OcrTask(**data)
        except ValidationError as exc:
            raise PermanentTaskError(f"Invalid OCR task payload: {exc}") from exc

        if not (task.has_pdf() or task.has_image()):
            raise PermanentTaskError(
                "Payload must contain one of: image_url, image_base64, image_path, "
                "or pdf_bucket+pdf_key"
            )
        return task

    # image tasks

    async def _process_image(self, task: OcrTask) -> str:
        image = await self._load_image(task)
        image = await asyncio.to_thread(self._skew_corrector.correct, image)
        return (await self._predict([image]))[0]

    async def _load_image(self, task: OcrTask) -> Image.Image:
        """Load the task's image as RGB. An unreadable image is a permanent error."""
        try:
            image = await asyncio.to_thread(self._image_loader.load, task.model_dump())
        except (ValueError, UnidentifiedImageError) as exc:
            # Bad base64, unknown format, or a path outside the allowed root:
            # the same payload fails every time. Network errors are not caught,
            # so they are retried.
            raise PermanentTaskError(f"Unreadable image: {exc}") from exc

        try:
            # Skew correction fails on gray, palette and 1-bit images, and the
            # PNG data URI fails on CMYK. convert() also decodes the pixels, so
            # a truncated file fails here.
            return await asyncio.to_thread(image.convert, "RGB")
        except OSError as exc:
            raise PermanentTaskError(f"Corrupt image: {exc}") from exc

    # PDF tasks

    async def _process_pdf(self, task: OcrTask) -> str:
        """Download the PDF and return its text: the text layer plus the OCR of scans.

        One task is one PDF, handled by this worker. More throughput comes from
        more worker replicas taking other documents, not from splitting one
        PDF across workers. A PDF over `ocr_max_pdf_pages` is a permanent
        failure, so one huge document cannot hold the worker.
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

    async def _ocr_chunks(self, pdf_path: Path, cached: dict[int, str]) -> list[str]:
        """Return the PDF's text chunks in order, with the scanned ones OCR-ed.

        Scanned pages go to the model in batches of `ocr_page_batch_size`. CPU
        and GPU work overlap: while one batch is on the model, this loop renders
        and straightens the pages of the next one. At most one batch is on the
        model and one is being prepared, so memory stays bounded.

        Pages in `cached` (from a failed earlier run) are not rendered or
        OCR-ed again. Each finished batch is written to the checkpoint.
        """
        batch_size = self.cfg.ocr_page_batch_size
        chunks: list[str | None] = []  # None marks a scanned page not yet OCR-ed
        pending: list[_ScannedPage] = []  # prepared pages, waiting for a full batch
        in_flight: _Batch | None = None  # the batch on the model now
        next_number = 0  # scanned-chunk counter, in the extractor's order

        try:
            async for kind, content, tag in self._pdf_extractor.extract_chunks(
                pdf_path, cached_images=cached.keys()
            ):
                if kind == "text":
                    chunks.append(content)
                    continue

                number, next_number = next_number, next_number + 1
                if kind == "cached":
                    chunks.append(self._with_tag(tag, cached[number]))
                    continue

                image = await asyncio.to_thread(self._skew_corrector.correct, content)
                chunks.append(None)
                pending.append(_ScannedPage(len(chunks) - 1, number, image, tag))
                if len(pending) >= batch_size:
                    # The previous batch has had the whole preparation of this
                    # one to run. Collect it, then start this one.
                    await self._finish_batch(in_flight, chunks)
                    in_flight = self._start_batch(pending)
                    pending = []

            await self._finish_batch(in_flight, chunks)
            in_flight = None
            if pending:  # the last, smaller batch
                in_flight = self._start_batch(pending)
                await self._finish_batch(in_flight, chunks)
                in_flight = None
        finally:
            # An error while preparing pages must not abandon a running model
            # call: wait for it and drop its result.
            if in_flight is not None and not in_flight.task.done():
                await asyncio.gather(in_flight.task, return_exceptions=True)

        return [text for text in chunks if text is not None]

    def _start_batch(self, pages: list[_ScannedPage]) -> _Batch:
        """Start the model call for `pages` without waiting for it."""
        task = asyncio.create_task(self._predict([page.image for page in pages]))
        return _Batch(task, pages)

    async def _finish_batch(self, batch: _Batch | None, chunks: list[str | None]) -> None:
        """Wait for a started batch, fill its slots in `chunks`, and checkpoint it."""
        if batch is None:
            return

        texts = await batch.task
        for page, text in zip(batch.pages, texts):
            chunks[page.slot] = self._with_tag(page.tag, text)

        if self._checkpoint is not None:
            await self._checkpoint.save(
                {page.number: text for page, text in zip(batch.pages, texts)}
            )

    @staticmethod
    def _with_tag(tag: str, text: str) -> str:
        """Put the scanned-page marker, if any, on its own line before the text."""
        return f"\n{tag}\n{text}" if tag else text

    # checkpoint

    async def _open_checkpoint(self, task: OcrTask, pdf_bytes: bytes) -> dict[int, str]:
        """Set up the checkpoint of this PDF and return the pages it already has.

        Returns {} when the checkpoint is off (no Redis, or a TTL of 0).
        """
        if self.redis is None or self.cfg.ocr_partial_ttl_seconds == 0:
            return {}

        key = await asyncio.to_thread(
            checkpoint_key, self.stream, task.document_id, pdf_bytes, self._ocr_settings_tag()
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

    def _ocr_settings_tag(self) -> str:
        """The settings that change the OCR text. They are part of the checkpoint key.

        A saved page is reused only if all of them are unchanged.
        """
        return "|".join(
            str(value)
            for value in (
                self.cfg.ocr_model_name,
                self.cfg.ocr_model_uri,
                self.cfg.ocr_render_dpi,
                self.cfg.ocr_max_tokens,
            )
        )

    async def _clear_checkpoint(self) -> None:
        if self._checkpoint is not None:
            await self._checkpoint.clear()

    # model calls

    async def _predict(self, images: list[Image.Image]) -> list[str]:
        """Run one model call for `images`, bounded by a time limit.

        A Python thread cannot be cancelled, so after a timeout the call may
        still run on the GPU. Reusing the model would put two calls on one
        engine, which is not thread-safe. So a timeout fails this message (the
        normal retry/DLQ path counts it) and retires the worker: it stops
        reading, and run() exits the process for the restart policy to replace.
        """
        try:
            texts = await asyncio.wait_for(
                asyncio.to_thread(self._model.predict_batch, images, self.cfg.ocr_max_tokens),
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
            raise RuntimeError(f"Model returned {len(texts)} texts for {len(images)} images")
        return texts

    # model and client set-up

    async def _ensure_model(self) -> None:
        """Load the model on the first task. Loading runs in a thread."""
        if self._model is None:
            await asyncio.to_thread(self._load_model)

    def _load_model(self) -> None:
        self._model = create_model(
            model_name=self.cfg.ocr_model_name,
            cache_dir=self.cfg.model_base_path,
            gpu_device=self.cfg.ocr_gpu_device,
            model_uri=self.cfg.ocr_model_uri,
            dtype=self.cfg.ocr_dtype,
            gpu_memory_utilization=self.cfg.ocr_gpu_memory_utilization,
            max_model_len=self.cfg.ocr_max_model_len,
            revision=revision_for(self.cfg.ocr_model_name, self.cfg),
        )
        self._model.load()
        self.logger.info(
            "Model loaded on %s: %s", self._model.device, type(self._model).__name__
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
                lineage=self._manifest.lineage(),
            )

    # output

    def _save_to_bronze(self, task: OcrTask, text: str) -> None:
        self._ensure_storage()
        if not self._manifest_published:
            publish_manifest(self._storage, self._manifest)
            self._manifest_published = True
        self._storage.save(
            prefix="ocr",
            document_id=task.document_id,
            data=text.encode("utf-8"),
            metadata=self._manifest.lineage(),
        )

    async def _publish_result(self, task: OcrTask, text: str) -> None:
        self._ensure_publisher()
        if self._result_publisher is not None:
            await self._result_publisher.publish(task.model_dump(), text)
