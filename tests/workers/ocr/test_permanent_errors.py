"""Tasks that can never succeed go straight to the DLQ (ADR-011)."""
from unittest.mock import patch

import pytest
from PIL import UnidentifiedImageError

from workers.base_worker import PermanentTaskError
from workers.ocr_worker import OcrWorker


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
