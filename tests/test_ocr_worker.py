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
        """process deve carregar imagem, preprocessar, predizer e publicar resultado."""
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
        """process com payload sem fonte de imagem deve levantar ValueError."""
        with pytest.raises(ValueError, match="image_url"):
            await worker.process({"document_id": "no-image"})

    async def test_unload_on_shutdown(self, mock_model):
        """run deve chamar unload do modelo no finally."""
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
        """process com pdf_bucket/pdf_key deve baixar, dividir em chunks e OCRar as imagens."""
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
        """PDF com mais páginas que ocr_max_pdf_pages deve levantar antes de extrair."""
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
        """Se ambos pdf_* e image_* vierem no payload, o caminho PDF vence."""
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
        """process deve carregar modelo automaticamente se None."""
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
        """MinIO fora do ar: a task falha (retry/DLQ) e nada é publicado. Antes
        ela era acked com o texto publicado em results mas nunca gravado no
        bronze — gap permanente e silencioso."""
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
