"""Scanned pages go to the model in bounded batches, and rendering overlaps the model call (ADR-012)."""
import threading
from unittest.mock import patch

import pytest
from PIL import Image

from config import Settings
from tests.workers.ocr.helpers import FakeRedis, PDF_TASK, run_pdf, scanned_pdf


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
        storage, page_count, extract, skew = run_pdf(worker, chunks)

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

        with scanned_pdf(
            worker, pages=4, on_yield=lambda n: third_page_ready.set() if n == 2 else None
        ):
            worker.redis = FakeRedis()
            await worker.process(PDF_TASK)

        assert results["first"] is True
