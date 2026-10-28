from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from workers.ocr_worker import OcrWorker


@pytest.fixture
def mock_model():
    """Retorna um model mockado que implementa AbstractOcrModel."""
    model = MagicMock()
    model.device = "cpu"
    model.predict.return_value = "texto extraído via OCR"
    return model


@pytest.fixture
def worker(mock_redis, mock_model):
    """OcrWorker com Redis mocado, model injetado e MinIO mockado."""
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


class TestOcrWorker:
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
            from PIL import Image
            fake_img = Image.new("RGB", (100, 50), color="white")
            mock_load.return_value = fake_img
            mock_skew.return_value = fake_img

            await worker.process(ocr_payload)

            # Verifica que predict foi chamado com a imagem e max_tokens correto
            mock_model.predict.assert_called_once()
            args, kwargs = mock_model.predict.call_args
            assert args[0].size == (100, 50)
            assert args[1] == worker.cfg.ocr_max_tokens

            # Verifica que o resultado foi publicado no Redis
            mock_redis.xadd.assert_awaited_once()
            call_args = mock_redis.xadd.call_args[0]
            assert "results" in call_args[0]
            assert call_args[1]["document_id"] == "test-001"
            assert call_args[1]["text"] == "texto extraído via OCR"

    async def test_process_no_image_raises(self, worker):
        """Arrange: a payload with no image or PDF source.
        Act: process it.
        Assert: ValueError naming image_url.
        """
        with pytest.raises(ValueError, match="image_url"):
            await worker.process({"document_id": "no-image"})

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

            # run() deve chamar unload_model mesmo com exceção
            with pytest.raises(Exception, match="force stop"):
                w.run()

            mock_model.unload.assert_called_once()

    async def test_process_pdf_combines_text_and_image_chunks(self, worker, mock_redis, mock_model):
        """Arrange: a PDF with one text chunk and one scanned-image chunk.
        Act: process the PDF payload.
        Assert: the PDF is downloaded, only the image chunk is OCRed, and the
        published text has the text layer, the OCR output, and the scanned-page
        marker in order.
        """
        async def fake_chunks(pdf_path):
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

    async def test_process_pdf_takes_priority_over_image_fields(self, worker, mock_redis, mock_model):
        """Arrange: a payload carrying both pdf_* and image_url.
        Act: process it.
        Assert: the PDF path wins — it is downloaded and the image URL is
        ignored.
        """
        async def fake_chunks(pdf_path):
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
            from PIL import Image
            mock_load.return_value = Image.new("RGB", (10, 10), color="white")
            mock_skew.return_value = Image.new("RGB", (10, 10), color="white")

            w = OcrWorker()
            w.redis = mock_redis
            # _model = None (não carregado)
            assert w._model is None

            await w.process({"document_id": "d1", "image_url": "http://example.com/img.png"})

            # Modelo deve ter sido carregado via create_model
            assert w._model is mock_model
            mock_model.predict.assert_called_once()

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
            from PIL import Image
            mock_load.return_value = Image.new("RGB", (10, 10), color="white")
            mock_skew.return_value = mock_load.return_value

            worker._ensure_storage()
            worker._storage.save.side_effect = Exception("minio down")

            with pytest.raises(Exception, match="minio down"):
                await worker.process(ocr_payload)

            mock_redis.xadd.assert_not_awaited()
