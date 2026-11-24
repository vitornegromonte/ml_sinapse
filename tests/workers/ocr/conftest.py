"""Fixtures for the OcrWorker tests."""
from unittest.mock import MagicMock, patch

import pytest

from workers.ocr_worker import OcrWorker


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
