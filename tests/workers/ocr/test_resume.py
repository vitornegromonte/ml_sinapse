"""A retry resumes from the Redis checkpoint of finished pages (ADR-012)."""
import threading
from unittest.mock import patch

import pytest

from config import Settings
from tests.workers.ocr.helpers import FakeRedis, PDF_TASK, TAG, echo_model, scanned_pdf


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
        redis = FakeRedis()
        worker.redis = redis
        first_run_images: list = []
        echo_model(mock_model, first_run_images, fail_on_call=2)
        with scanned_pdf(worker, pages=5):
            with pytest.raises(RuntimeError, match="gpu fault"):
                await worker.process(PDF_TASK)
        saved = next(iter(redis.hashes.values()))
        assert set(saved) == {"0", "1"}

        retry_images: list = []
        seen_cached: list = []
        echo_model(mock_model, retry_images)
        published: dict = {}
        original_xadd = redis.xadd

        async def capture_xadd(stream, fields, **kwargs):
            published.update(fields)
            return await original_xadd(stream, fields, **kwargs)

        redis.xadd = capture_xadd
        with scanned_pdf(worker, pages=5, seen_cached=seen_cached):
            await worker.process(PDF_TASK)

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
        redis = FakeRedis(fail_xadd=True)
        worker.redis = redis
        echo_model(mock_model, [])

        with scanned_pdf(worker, pages=2):
            with pytest.raises(ConnectionError):
                await worker.process(PDF_TASK)

        assert len(redis.hashes) == 1
        assert redis.deleted == []

    async def test_checkpoint_is_cleared_after_a_successful_publish(self, worker, mock_model):
        """Arrange: a PDF that OCRs and publishes without error.
        Act: process the task.
        Assert: the checkpoint was deleted at the end.
        """
        redis = FakeRedis()
        worker.redis = redis
        echo_model(mock_model, [])

        with scanned_pdf(worker, pages=2):
            await worker.process(PDF_TASK)

        assert len(redis.deleted) == 1
        assert redis.hashes == {}

    async def test_ttl_zero_turns_the_checkpoint_off(self, worker, mock_model):
        """Arrange: ocr_partial_ttl_seconds=0 and a failing second batch.
        Act: process the task.
        Assert: nothing was written to Redis.
        """
        worker.cfg = Settings(_env_file=None, ocr_page_batch_size=1, ocr_partial_ttl_seconds=0)
        redis = FakeRedis()
        worker.redis = redis
        echo_model(mock_model, [], fail_on_call=2)

        with scanned_pdf(worker, pages=3):
            with pytest.raises(RuntimeError):
                await worker.process(PDF_TASK)

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
        worker.redis = FakeRedis()
        release = threading.Event()
        finished = threading.Event()

        def predict_batch(images, max_tokens, prompt=None):
            release.wait(timeout=5)
            finished.set()
            return [f"OCR:{image}" for image in images]

        mock_model.predict_batch.side_effect = predict_batch

        async def broken_chunks(pdf_path, cached_images=()):
            yield ("image", "img0", TAG)
            yield ("image", "img1", TAG)
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
                await worker.process(PDF_TASK)

        assert finished.is_set()
