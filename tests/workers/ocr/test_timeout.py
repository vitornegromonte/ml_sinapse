"""A model call that times out retires the worker (ADR-011)."""
import threading
from unittest.mock import patch

import pytest
from PIL import Image

from config import Settings
from workers.ocr_worker import OcrTimeoutError, OcrWorker


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
