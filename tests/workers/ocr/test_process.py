"""The main flow of a task: image and PDF input, the result, and the model lifecycle."""
from unittest.mock import patch

import pytest
from PIL import Image

from config import Settings
from tests.workers.ocr.helpers import run_pdf
from workers.ocr_worker import OcrWorker


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
        storage, page_count, extract, skew = run_pdf(worker, [("text", "so texto", "")])

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


class TestOcrWorkerRevisionPin:
    def test_load_model_passes_the_pinned_revision(self, tmp_path):
        """Arrange: a lock that pins the OCR model, and create_model patched.
        Act: load the model.
        Assert: create_model gets that revision.
        """
        import json

        lock = tmp_path / "lock.json"
        lock.write_text(json.dumps({
            "lock_version": 1,
            "huggingface": {"Dharma-AI/Dharma-OCR-LITE": {"revision": "f" * 40}},
        }))
        cfg = Settings(_env_file=None, model_lock_path=str(lock))
        with patch("workers.ocr_worker.create_model") as mock_create:
            OcrWorker(cfg)._load_model()

        assert mock_create.call_args.kwargs["revision"] == "f" * 40

    def test_load_model_without_a_pin_passes_none(self, tmp_path):
        """Arrange: a lock that pins nothing.
        Act: load the model.
        Assert: create_model gets revision=None.
        """
        import json

        lock = tmp_path / "empty.json"
        lock.write_text(json.dumps({"lock_version": 1}))
        cfg = Settings(_env_file=None, model_lock_path=str(lock))
        with patch("workers.ocr_worker.create_model") as mock_create:
            OcrWorker(cfg)._load_model()

        assert mock_create.call_args.kwargs["revision"] is None


class TestOcrWorkerLineage:
    async def _run_two_tasks(self, worker):
        with patch("workers.ocr_worker.ImageLoader.load") as mock_load:
            mock_load.return_value = Image.new("RGB", (100, 50))
            await worker.process({"document_id": "d1", "image_base64": "x"})
            await worker.process({"document_id": "d2", "image_base64": "x"})

    async def test_the_result_message_carries_the_manifest(self, worker, mock_redis):
        """Arrange: an OCR worker.
        Act: process an image task.
        Assert: the message on the results stream has the worker's manifest_id and model.
        """
        await self._run_two_tasks(worker)

        fields = mock_redis.xadd.call_args_list[0].args[1]
        assert fields["manifest_id"] == worker._manifest.manifest_id
        assert fields["model"] == worker._manifest.model

    async def test_the_bronze_object_carries_the_manifest_as_metadata(self, worker):
        """Arrange: an OCR worker.
        Act: process an image task.
        Assert: the text is saved to bronze with the lineage as object metadata.
        """
        await self._run_two_tasks(worker)

        saves = [c for c in worker._storage.save.call_args_list if c.kwargs["prefix"] == "ocr"]
        assert saves[0].kwargs["metadata"] == worker._manifest.lineage()

    async def test_the_manifest_document_is_published_once(self, worker):
        """Arrange: an OCR worker.
        Act: process two tasks.
        Assert: the manifest document is written to bronze one time, not per task.
        """
        await self._run_two_tasks(worker)

        manifests = [c for c in worker._storage.save.call_args_list if c.kwargs["prefix"] == "manifests"]
        assert len(manifests) == 1
        assert manifests[0].kwargs["document_id"] == worker._manifest.manifest_id

    def test_the_manifest_names_the_configured_model_and_its_pin(self, worker):
        """Arrange: a worker with the default settings.
        Act: read its manifest.
        Assert: Dharma-OCR-LITE is named, with the settings that shape its output.
        """
        identity = worker._manifest.identity

        assert identity["model"] == "Dharma-AI/Dharma-OCR-LITE"
        assert identity["render_dpi"] == 200
        assert identity["repetition_penalty"] == 1.05
