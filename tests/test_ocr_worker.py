"""Tests for OcrWorker (workers/ocr_worker.py).

The model, Redis, MinIO and the PDF extractor are mocked, so no GPU or network
is needed. Layout: shared fixtures and helpers first, then one class per
behavior, in the order a task meets them.
"""
import threading
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image, UnidentifiedImageError

from config import Settings
from workers.base_worker import PermanentTaskError
from workers.ocr_worker import OcrTimeoutError, OcrWorker


# fixtures


@pytest.fixture
def mock_model():
    """A mocked OCR model that implements AbstractOcrModel."""
    model = MagicMock()
    model.device = "cpu"
    model.predict.return_value = "texto extraído via OCR"
    # Same as AbstractOcrModel.predict_batch: one predict() per image.
    model.predict_batch.side_effect = lambda images, max_tokens, prompt=None: [
        model.predict(image, max_tokens, prompt) for image in images
    ]
    return model


@pytest.fixture
def worker(mock_redis, mock_model):
    """An OcrWorker with Redis and MinIO mocked and the model already injected."""
    with (
        patch("workers.ocr_worker.create_model") as mock_create,
        patch("workers.ocr_worker.StorageService") as mock_storage_cls,
    ):
        mock_create.return_value = mock_model
        mock_storage_cls.return_value.save.return_value = None

        w = OcrWorker()
        w.redis = mock_redis
        w._model = mock_model
        yield w


# helpers


class _FakeRedis:
    """In-memory Redis: just the calls OcrWorker, ResultPublisher and the
    checkpoint make. `fail_xadd` makes the result publish fail."""

    def __init__(self, fail_xadd: bool = False):
        self.hashes: dict[str, dict[str, str]] = {}
        self.deleted: list[str] = []
        self._fail_xadd = fail_xadd

    async def xadd(self, *args, **kwargs):
        if self._fail_xadd:
            raise ConnectionError("redis down")
        return "1-0"

    async def xlen(self, *args, **kwargs):
        return 0

    async def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    async def hset(self, key, mapping):
        self.hashes.setdefault(key, {}).update(mapping)

    async def expire(self, key, ttl):
        pass

    async def delete(self, key):
        self.hashes.pop(key, None)
        self.deleted.append(key)


_TAG = "[Anexo Escaneado Intercalado]"
_PDF_TASK = {"document_id": "doc-r", "pdf_bucket": "bronze", "pdf_key": "k.pdf"}


@contextmanager
def _scanned_pdf(worker, pages, seen_cached=None, on_yield=None):
    """Make a worker see a 'PDF' of one text chunk and `pages` scanned pages.
    The fake extractor honors cached_images like the real PdfExtractor."""

    async def fake_chunks(pdf_path, cached_images=()):
        if seen_cached is not None:
            seen_cached.append(set(cached_images))
        yield ("text", "inicio", "")
        for number in range(pages):
            if on_yield is not None:
                on_yield(number)
            if number in cached_images:
                yield ("cached", "", _TAG)
            else:
                yield ("image", f"img{number}", _TAG)

    with (
        patch("workers.ocr_worker.StorageService") as storage_cls,
        patch.object(worker._pdf_extractor, "page_count", return_value=pages),
        patch.object(worker._pdf_extractor, "extract_chunks", side_effect=fake_chunks),
        patch("workers.ocr_worker.SkewCorrector.correct", side_effect=lambda img: img),
    ):
        storage_cls.return_value.download.return_value = b"%PDF-same-bytes"
        worker._storage = storage_cls.return_value
        yield


def _echo_model(model, images_seen, fail_on_call=None):
    calls = {"n": 0}

    def predict_batch(images, max_tokens, prompt=None):
        calls["n"] += 1
        if fail_on_call == calls["n"]:
            raise RuntimeError("gpu fault")
        images_seen.append(list(images))
        return [f"OCR:{image}" for image in images]

    model.predict_batch.side_effect = predict_batch


def _run_pdf(worker, chunks):
    """Four patches that make `worker` see a PDF made of exactly `chunks`.

    Enter them all with `with`. The fake extractor ignores `cached_images`.
    """

    async def fake_chunks(pdf_path, cached_images=()):
        for chunk in chunks:
            yield chunk

    return (
        patch("workers.ocr_worker.StorageService"),
        patch.object(worker._pdf_extractor, "page_count", return_value=len(chunks)),
        patch.object(worker._pdf_extractor, "extract_chunks", side_effect=fake_chunks),
        patch("workers.ocr_worker.SkewCorrector.correct", side_effect=lambda img: img),
    )


# tests


class TestOcrWorkerConstruction:
    """Building the worker from settings."""

    def test_pdf_extractor_uses_configured_dpi(self, mock_redis):
        """Arrange: settings with a custom ocr_render_dpi.
        Act: build the worker.
        Assert: the PDF extractor renders at that DPI.
        """
        cfg = Settings(_env_file=None, ocr_render_dpi=300)

        worker = OcrWorker(cfg)

        assert worker._pdf_extractor._dpi == 300


class TestOcrWorkerImageTasks:
    """Image tasks: load, convert to RGB, straighten, OCR, publish."""

    async def test_process_success(self, worker, mock_redis, mock_model, ocr_payload):
        """Arrange: a worker with image loading and skew correction patched.
        Act: process an image payload.
        Assert: predict gets the corrected image and configured max tokens, and
        the OCR text is published to the results stream.
        """
        with (
            patch("workers.ocr_worker.ImageLoader.load") as mock_load,
            patch("workers.ocr_worker.SkewCorrector.correct") as mock_skew,
        ):
            fake_img = Image.new("RGB", (100, 50), color="white")
            mock_load.return_value = fake_img
            mock_skew.return_value = fake_img

            await worker.process(ocr_payload)

            # predict was called with the image and the configured max_tokens
            mock_model.predict.assert_called_once()
            args, kwargs = mock_model.predict.call_args
            assert args[0].size == (100, 50)
            assert args[1] == worker.cfg.ocr_max_tokens

            # the result was published to Redis
            mock_redis.xadd.assert_awaited_once()
            call_args = mock_redis.xadd.call_args[0]
            assert "results" in call_args[0]
            assert call_args[1]["document_id"] == "test-001"
            assert call_args[1]["text"] == "texto extraído via OCR"

    @pytest.mark.parametrize("mode", ["L", "P", "1", "RGBA", "CMYK"])
    async def test_process_image_converts_any_mode_to_rgb(self, worker, mock_model, mode):
        """Arrange: an image in a non-RGB mode, real skew correction.
        Act: process an image task.
        Assert: predict receives an RGB image and the task succeeds.
        """
        with patch("workers.ocr_worker.ImageLoader.load") as mock_load:
            mock_load.return_value = Image.new(mode, (200, 200))

            await worker.process({"document_id": "d-1", "image_base64": "x"})

            sent_image = mock_model.predict.call_args[0][0]
            assert sent_image.mode == "RGB"


class TestOcrWorkerPdfTasks:
    """PDF tasks: text layer plus OCR of the scanned parts."""

    async def test_process_pdf_combines_text_and_image_chunks(self, worker, mock_redis, mock_model):
        """Arrange: a PDF with one text chunk and one scanned-image chunk.
        Act: process the PDF payload.
        Assert: the PDF is downloaded, only the image chunk is OCRed, and the
        published text has the text layer, the OCR output, and the scanned-page
        marker in order.
        """
        async def fake_chunks(pdf_path, cached_images=()):
            yield ("text", "texto direto da página 1", "")
            yield ("image", object(), "[Página Escaneada Integralmente]")

        with (
            patch("workers.ocr_worker.StorageService") as mock_storage_cls,
            patch.object(worker._pdf_extractor, "page_count", return_value=2),
            patch.object(worker._pdf_extractor, "extract_chunks", side_effect=fake_chunks),
            patch("workers.ocr_worker.SkewCorrector.correct", side_effect=lambda img: img),
        ):
            mock_storage_cls.return_value.download.return_value = b"%PDF-1.4 fake"
            worker._storage = mock_storage_cls.return_value

            await worker.process({
                "document_id": "inquerito-270080",
                "pdf_bucket": "bronze",
                "pdf_key": "inqueritos/pdf/270080_teste.pdf",
            })

            mock_storage_cls.return_value.download.assert_called_once_with(
                "bronze", "inqueritos/pdf/270080_teste.pdf"
            )
            mock_model.predict.assert_called_once()  # only the image chunk needs OCR

            call_args = mock_redis.xadd.call_args[0]
            published_text = call_args[1]["text"]
            assert "texto direto da página 1" in published_text
            assert "texto extraído via OCR" in published_text  # mock_model.predict's return
            assert "[Página Escaneada Integralmente]" in published_text

    async def test_process_pdf_takes_priority_over_image_fields(self, worker, mock_redis, mock_model):
        """Arrange: a payload carrying both pdf_* and image_url.
        Act: process it.
        Assert: the PDF path wins — it is downloaded and the image URL is
        ignored.
        """
        async def fake_chunks(pdf_path, cached_images=()):
            yield ("text", "só texto, sem imagens", "")

        with (
            patch("workers.ocr_worker.StorageService") as mock_storage_cls,
            patch.object(worker._pdf_extractor, "page_count", return_value=1),
            patch.object(worker._pdf_extractor, "extract_chunks", side_effect=fake_chunks),
        ):
            mock_storage_cls.return_value.download.return_value = b"%PDF-1.4 fake"
            worker._storage = mock_storage_cls.return_value

            await worker.process({
                "document_id": "d1",
                "pdf_bucket": "bronze",
                "pdf_key": "some.pdf",
                "image_url": "http://example.com/should-be-ignored.png",
            })

            mock_storage_cls.return_value.download.assert_called_once()
            mock_model.predict.assert_not_called()  # no image chunks in this fake PDF

    async def test_pdf_without_scanned_pages_makes_no_model_call(
        self, worker, mock_model
    ):
        """Arrange: a PDF with only text chunks.
        Act: process the PDF.
        Assert: predict_batch is never called.
        """
        storage, page_count, extract, skew = _run_pdf(worker, [("text", "so texto", "")])

        with storage as storage_cls, page_count, extract, skew:
            storage_cls.return_value.download.return_value = b"%PDF"
            worker._storage = storage_cls.return_value

            await worker.process(
                {"document_id": "d-t", "pdf_bucket": "bronze", "pdf_key": "k.pdf"}
            )

        mock_model.predict_batch.assert_not_called()

    async def test_process_pdf_rejects_over_page_cap(self, worker):
        """Arrange: a 999-page PDF with the cap at 200.
        Act: process it.
        Assert: ValueError naming 999 pages and extraction is never attempted.
        """
        with (
            patch("workers.ocr_worker.StorageService") as mock_storage_cls,
            patch.object(worker._pdf_extractor, "page_count", return_value=999),
            patch.object(worker._pdf_extractor, "extract_chunks") as mock_extract,
        ):
            mock_storage_cls.return_value.download.return_value = b"%PDF-1.4 fake"
            worker._storage = mock_storage_cls.return_value
            worker.cfg.ocr_max_pdf_pages = 200

            with pytest.raises(ValueError, match="999 pages"):
                await worker.process({
                    "document_id": "huge-doc",
                    "pdf_bucket": "bronze",
                    "pdf_key": "inqueritos/pdf/huge.pdf",
                })

            mock_extract.assert_not_called()  # rejected before paying for extraction


class TestOcrWorkerResult:
    """What happens to the OCR text: validation, logging, bronze, publish."""

    async def test_process_empty_text_raises(self, worker, mock_redis, mock_model):
        """Arrange: a model that returns only whitespace.
        Act: process an image task.
        Assert: ValueError is raised and nothing is published.
        """
        mock_model.predict.return_value = "   \n"
        with patch("workers.ocr_worker.ImageLoader.load") as mock_load:
            mock_load.return_value = Image.new("RGB", (100, 50))

            with pytest.raises(ValueError, match="no text"):
                await worker.process({"document_id": "d-2", "image_base64": "x"})

        mock_redis.xadd.assert_not_awaited()

    async def test_process_does_not_log_ocr_text(self, worker, caplog):
        """Arrange: a model that returns a recognizable text.
        Act: process an image task with log capture on.
        Assert: the text is absent from the logs; the length is present.
        """
        secret = "texto-sensivel-do-bo"
        worker._model.predict.return_value = secret
        with patch("workers.ocr_worker.ImageLoader.load") as mock_load:
            mock_load.return_value = Image.new("RGB", (100, 50))

            with caplog.at_level("INFO"):
                await worker.process({"document_id": "d-3", "image_base64": "x"})

        assert secret not in caplog.text
        assert f"text_length={len(secret)}" in caplog.text

    async def test_save_failure_fails_before_publishing(
        self, worker, mock_redis, mock_model, ocr_payload
    ):
        """Arrange: a storage service whose bronze save fails.
        Act: process one image payload.
        Assert: the error propagates and nothing is published — before, the task
        was acked with the text in results but never in bronze, a silent
        permanent gap.
        """
        with (
            patch("workers.ocr_worker.ImageLoader.load") as mock_load,
            patch("workers.ocr_worker.SkewCorrector.correct") as mock_skew,
        ):
            mock_load.return_value = Image.new("RGB", (10, 10), color="white")
            mock_skew.return_value = mock_load.return_value

            worker._ensure_storage()
            worker._storage.save.side_effect = Exception("minio down")

            with pytest.raises(Exception, match="minio down"):
                await worker.process(ocr_payload)

            mock_redis.xadd.assert_not_awaited()


class TestOcrWorkerModelLifecycle:
    """Lazy model load and unload at shutdown."""

    async def test_model_loaded_lazily(self, mock_redis, mock_model):
        """Arrange: a new worker whose model and storage are still None.
        Act: process the first image payload.
        Assert: create_model builds the model and predict runs once.
        """
        with (
            patch("workers.ocr_worker.create_model") as mock_create,
            patch("workers.ocr_worker.ImageLoader.load") as mock_load,
            patch("workers.ocr_worker.SkewCorrector.correct") as mock_skew,
            # O bronze agora propaga falha: sem este patch o teste tentava
            # falar com um MinIO real e passava só porque o erro era engolido.
            patch("workers.ocr_worker.StorageService"),
        ):
            mock_create.return_value = mock_model
            mock_load.return_value = Image.new("RGB", (10, 10), color="white")
            mock_skew.return_value = Image.new("RGB", (10, 10), color="white")

            w = OcrWorker()
            w.redis = mock_redis
            # _model = None (não carregado)
            assert w._model is None

            await w.process({"document_id": "d1", "image_url": "http://example.com/img.png"})

            # the model was loaded through create_model
            assert w._model is mock_model
            mock_model.predict.assert_called_once()

    async def test_unload_on_shutdown(self, mock_model):
        """Arrange: a worker with an injected model and run() forced to raise.
        Act: run the worker.
        Assert: the exception propagates and the model is unloaded in finally.
        """
        with (
            patch("workers.ocr_worker.create_model") as mock_create,
            patch("workers.ocr_worker.BaseWorker.run") as mock_run,
        ):
            mock_create.return_value = mock_model
            mock_run.side_effect = Exception("force stop")

            w = OcrWorker()
            w._model = mock_model

            # run() unloads the model even when the base run() raises
            with pytest.raises(Exception, match="force stop"):
                w.run()

            mock_model.unload.assert_called_once()


class TestOcrWorkerPermanentErrors:
    """Tasks that can never succeed go straight to the DLQ (ADR-011)."""

    async def test_process_no_image_raises(self, worker):
        """Arrange: a payload with no image or PDF source.
        Act: process it.
        Assert: ValueError naming image_url.
        """
        with pytest.raises(ValueError, match="image_url"):
            await worker.process({"document_id": "no-image"})

    async def test_invalid_payload_is_permanent_and_skips_model_load(self, mock_redis):
        """Arrange: a worker without a loaded model and a payload of wrong types.
        Act: process it.
        Assert: PermanentTaskError, and the model is never loaded.
        """
        with patch("workers.ocr_worker.create_model") as mock_create:
            w = OcrWorker()
            w.redis = mock_redis

            with pytest.raises(PermanentTaskError, match="Invalid OCR task"):
                await w.process({"document_id": "d", "pdf_bucket": ["not", "a", "str"]})

        mock_create.assert_not_called()

    async def test_payload_without_source_is_permanent(self, worker):
        """Arrange: a payload with no image or PDF source.
        Act: process it.
        Assert: PermanentTaskError.
        """
        with pytest.raises(PermanentTaskError):
            await worker.process({"document_id": "d"})

    async def test_unreadable_image_is_permanent(self, worker):
        """Arrange: an image loader that rejects the payload.
        Act: process an image task.
        Assert: PermanentTaskError that keeps the original cause.
        """
        with patch("workers.ocr_worker.ImageLoader.load") as mock_load:
            mock_load.side_effect = UnidentifiedImageError("cannot identify image file")

            with pytest.raises(PermanentTaskError, match="Unreadable image") as info:
                await worker.process({"document_id": "d", "image_base64": "x"})

        assert isinstance(info.value.__cause__, UnidentifiedImageError)

    async def test_network_error_while_loading_image_is_not_permanent(self, worker):
        """Arrange: an image loader that fails with a connection error.
        Act: process an image task.
        Assert: the error is NOT a PermanentTaskError, so it can be retried.
        """
        with patch("workers.ocr_worker.ImageLoader.load") as mock_load:
            mock_load.side_effect = ConnectionError("down")

            with pytest.raises(ConnectionError):
                await worker.process({"document_id": "d", "image_url": "http://x"})

    async def test_pdf_over_page_cap_is_permanent(self, worker):
        """Arrange: a PDF whose page count is over ocr_page cap.
        Act: process it.
        Assert: PermanentTaskError.
        """
        with (
            patch("workers.ocr_worker.StorageService") as storage_cls,
            patch.object(worker._pdf_extractor, "page_count", return_value=10_000),
        ):
            storage_cls.return_value.download.return_value = b"%PDF"
            worker._storage = storage_cls.return_value

            with pytest.raises(PermanentTaskError, match="exceeding"):
                await worker.process(
                    {"document_id": "d", "pdf_bucket": "b", "pdf_key": "k.pdf"}
                )


class TestOcrWorkerBatching:
    """Scanned pages go to the model in bounded batches (ADR-012)."""

    async def test_pdf_pages_go_to_the_model_in_bounded_batches_in_order(
        self, worker, mock_redis, mock_model
    ):
        """Arrange: batch size 2 and a PDF of text, 5 scanned pages, and text.
        Act: process the PDF.
        Assert: the model gets batches of 2, 2 and 1 pages, and the published
        text keeps the PDF order.
        """
        worker.cfg = Settings(_env_file=None, ocr_page_batch_size=2)
        mock_model.predict_batch.side_effect = lambda images, max_tokens, prompt=None: [
            f"OCR:{image}" for image in images
        ]
        chunks = [("text", "inicio", "")]
        chunks += [("image", f"p{n}", "[P]") for n in range(5)]
        chunks += [("text", "fim", "")]
        storage, page_count, extract, skew = _run_pdf(worker, chunks)

        with storage as storage_cls, page_count, extract, skew:
            storage_cls.return_value.download.return_value = b"%PDF"
            worker._storage = storage_cls.return_value

            await worker.process(
                {"document_id": "d-b", "pdf_bucket": "bronze", "pdf_key": "k.pdf"}
            )

        batch_sizes = [len(c[0][0]) for c in mock_model.predict_batch.call_args_list]
        published = mock_redis.xadd.call_args[0][1]["text"]
        assert batch_sizes == [2, 2, 1]
        order = [published.index(x) for x in ("inicio", "OCR:p0", "OCR:p1", "OCR:p2", "OCR:p3", "OCR:p4", "fim")]
        assert order == sorted(order)

    async def test_wrong_result_count_raises(self, worker, mock_model):
        """Arrange: a model that returns fewer texts than images.
        Act: process an image task.
        Assert: RuntimeError, so a short result is never published.
        """
        mock_model.predict_batch.side_effect = lambda images, max_tokens, prompt=None: []
        with patch("workers.ocr_worker.ImageLoader.load") as mock_load:
            mock_load.return_value = Image.new("RGB", (10, 10))

            with pytest.raises(RuntimeError, match="0 texts for 1 images"):
                await worker.process({"document_id": "d", "image_base64": "x"})


class TestOcrWorkerOverlap:
    """Rendering the next batch overlaps the model call (ADR-012)."""

    async def test_next_batch_is_prepared_while_the_model_runs(self, worker, mock_model):
        """Arrange: batch size 2, a 4-page scan, and a model call that blocks
        until the extractor has produced page 3 (a threading.Event).
        Act: process the PDF.
        Assert: the event was set while the first model call was still running,
        so rendering overlapped with the GPU. Without overlap the call would
        wait out its 5 s limit and report False.
        """
        worker.cfg = Settings(_env_file=None, ocr_page_batch_size=2)
        third_page_ready = threading.Event()
        results = {}

        def predict_batch(images, max_tokens, prompt=None):
            if "first" not in results:
                results["first"] = third_page_ready.wait(timeout=5)
            return [f"OCR:{image}" for image in images]

        mock_model.predict_batch.side_effect = predict_batch

        with _scanned_pdf(
            worker, pages=4, on_yield=lambda n: third_page_ready.set() if n == 2 else None
        ):
            worker.redis = _FakeRedis()
            await worker.process(_PDF_TASK)

        assert results["first"] is True


class TestOcrWorkerResume:
    """A retry resumes from the Redis checkpoint (ADR-012)."""

    async def test_retry_ocrs_only_the_pages_that_never_finished(self, worker, mock_model):
        """Arrange: batch size 2, a 5-page scan, and a model that fails on its
        second call, so pages 0 and 1 finish and the task fails.
        Act: process the same task again with a working model.
        Assert: the retry sends only pages 2, 3 and 4 to the model, tells the
        extractor that pages 0 and 1 are cached, and publishes all five texts
        in order.
        """
        worker.cfg = Settings(_env_file=None, ocr_page_batch_size=2)
        redis = _FakeRedis()
        worker.redis = redis
        first_run_images: list = []
        _echo_model(mock_model, first_run_images, fail_on_call=2)
        with _scanned_pdf(worker, pages=5):
            with pytest.raises(RuntimeError, match="gpu fault"):
                await worker.process(_PDF_TASK)
        saved = next(iter(redis.hashes.values()))
        assert set(saved) == {"0", "1"}

        retry_images: list = []
        seen_cached: list = []
        _echo_model(mock_model, retry_images)
        published: dict = {}
        original_xadd = redis.xadd

        async def capture_xadd(stream, fields, **kwargs):
            published.update(fields)
            return await original_xadd(stream, fields, **kwargs)

        redis.xadd = capture_xadd
        with _scanned_pdf(worker, pages=5, seen_cached=seen_cached):
            await worker.process(_PDF_TASK)

        assert retry_images == [["img2", "img3"], ["img4"]]
        assert seen_cached == [{0, 1}]
        text = published["text"]
        positions = [text.index(f"OCR:img{n}") for n in range(5)]
        assert positions == sorted(positions)
        assert redis.hashes == {}

    async def test_checkpoint_is_kept_when_publishing_fails(self, worker, mock_model):
        """Arrange: a PDF whose OCR succeeds but whose result publish fails.
        Act: process the task.
        Assert: the error propagates and the checkpoint is still in Redis, so
        the retry does not repeat the OCR.
        """
        redis = _FakeRedis(fail_xadd=True)
        worker.redis = redis
        _echo_model(mock_model, [])

        with _scanned_pdf(worker, pages=2):
            with pytest.raises(ConnectionError):
                await worker.process(_PDF_TASK)

        assert len(redis.hashes) == 1
        assert redis.deleted == []

    async def test_checkpoint_is_cleared_after_a_successful_publish(self, worker, mock_model):
        """Arrange: a PDF that OCRs and publishes without error.
        Act: process the task.
        Assert: the checkpoint was deleted at the end.
        """
        redis = _FakeRedis()
        worker.redis = redis
        _echo_model(mock_model, [])

        with _scanned_pdf(worker, pages=2):
            await worker.process(_PDF_TASK)

        assert len(redis.deleted) == 1
        assert redis.hashes == {}

    async def test_ttl_zero_turns_the_checkpoint_off(self, worker, mock_model):
        """Arrange: ocr_partial_ttl_seconds=0 and a failing second batch.
        Act: process the task.
        Assert: nothing was written to Redis.
        """
        worker.cfg = Settings(_env_file=None, ocr_page_batch_size=1, ocr_partial_ttl_seconds=0)
        redis = _FakeRedis()
        worker.redis = redis
        _echo_model(mock_model, [], fail_on_call=2)

        with _scanned_pdf(worker, pages=3):
            with pytest.raises(RuntimeError):
                await worker.process(_PDF_TASK)

        assert redis.hashes == {}

    async def test_an_error_while_preparing_pages_waits_for_the_running_model_call(
        self, worker, mock_model
    ):
        """Arrange: batch size 1; the extractor fails on page 2 while the model
        is still busy with page 1 (a threading.Event holds it).
        Act: process the PDF.
        Assert: the extractor's error propagates, and the model call was waited
        for (not abandoned) before process() returned.
        """
        worker.cfg = Settings(_env_file=None, ocr_page_batch_size=1)
        worker.redis = _FakeRedis()
        release = threading.Event()
        finished = threading.Event()

        def predict_batch(images, max_tokens, prompt=None):
            release.wait(timeout=5)
            finished.set()
            return [f"OCR:{image}" for image in images]

        mock_model.predict_batch.side_effect = predict_batch

        async def broken_chunks(pdf_path, cached_images=()):
            yield ("image", "img0", _TAG)
            yield ("image", "img1", _TAG)
            release.set()
            raise OSError("fitz failed")

        with (
            patch("workers.ocr_worker.StorageService") as storage_cls,
            patch.object(worker._pdf_extractor, "page_count", return_value=3),
            patch.object(worker._pdf_extractor, "extract_chunks", side_effect=broken_chunks),
            patch("workers.ocr_worker.SkewCorrector.correct", side_effect=lambda img: img),
        ):
            storage_cls.return_value.download.return_value = b"%PDF"
            worker._storage = storage_cls.return_value

            with pytest.raises(OSError, match="fitz failed"):
                await worker.process(_PDF_TASK)

        assert finished.is_set()


class TestOcrWorkerTimeout:
    """A model call that times out retires the worker (ADR-011)."""

    async def test_timeout_fails_the_task_and_retires_the_worker(self, worker, mock_model):
        """Arrange: a model call that blocks longer than the 1 s timeout.
        Act: process an image task.
        Assert: OcrTimeoutError, the worker stops reading, and it is flagged
        so run() will not reuse or unload the model.
        """
        worker.cfg = Settings(_env_file=None, ocr_predict_timeout_seconds=1)
        release = threading.Event()
        mock_model.predict_batch.side_effect = lambda *a, **k: release.wait(10)
        try:
            with patch("workers.ocr_worker.ImageLoader.load") as mock_load:
                mock_load.return_value = Image.new("RGB", (10, 10))

                with pytest.raises(OcrTimeoutError):
                    await worker.process({"document_id": "d", "image_base64": "x"})
        finally:
            release.set()

        assert worker._running is False
        assert worker._model_stuck is True

    def test_run_exits_hard_without_unload_after_timeout(self, mock_model):
        """Arrange: a worker flagged as stuck and a base run() that returns.
        Act: run the worker.
        Assert: os._exit(1) is called and the model is not unloaded.
        """
        with (
            patch("workers.ocr_worker.BaseWorker.run"),
            patch("workers.ocr_worker.os._exit") as mock_exit,
        ):
            w = OcrWorker()
            w._model = mock_model
            w._model_stuck = True

            w.run()

        mock_exit.assert_called_once_with(1)
        mock_model.unload.assert_not_called()

    def test_run_unloads_model_when_not_stuck(self, mock_model):
        """Arrange: a healthy worker and a base run() that returns.
        Act: run the worker.
        Assert: the model is unloaded and the process is not force-exited.
        """
        with (
            patch("workers.ocr_worker.BaseWorker.run"),
            patch("workers.ocr_worker.os._exit") as mock_exit,
        ):
            w = OcrWorker()
            w._model = mock_model

            w.run()

        mock_model.unload.assert_called_once()
        mock_exit.assert_not_called()
