from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from workers.llm_worker import LlmWorker


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.predict.return_value = [{"text": "resumo gerado pelo LLM"}]
    return llm


@pytest.fixture
def worker(mock_redis, mock_llm):
    """LlmWorker com Redis mocado, LLM injetado e MinIO mockado."""
    with (
        patch("workers.llm_worker.LLMModel") as mock_llm_cls,
        patch("workers.llm_worker.StorageService") as mock_storage_cls,
    ):
        mock_llm_cls.return_value = mock_llm
        mock_storage_cls.return_value.save.return_value = None

        w = LlmWorker()
        w.redis = mock_redis
        w._llm = mock_llm
        yield w


@pytest.fixture
def llm_payload():
    return {"document_id": "bo-123", "inquiry_id": "42", "text": "texto extraído via OCR com 42 páginas..."}


class TestLlmWorker:
    async def test_process_success(self, worker, mock_redis, mock_llm, llm_payload):
        """Arrange: an LLM worker with mocked model and Redis, and a payload.
        Act: process it.
        Assert: one predict gets a prompt containing the text, and the summary is
        published to the LLM results stream.
        """
        await worker.process(llm_payload)

        mock_llm.predict.assert_called_once()
        args, _ = mock_llm.predict.call_args
        prompts = args[0]
        assert "texto extraído via OCR" in prompts[0]
        assert "Resumo" in prompts[0]

        mock_redis.xadd.assert_awaited_once()
        call_args = mock_redis.xadd.call_args[0]
        assert "llm:tasks" in call_args[0]
        assert call_args[1]["document_id"] == "bo-123"
        assert call_args[1]["text"] == "resumo gerado pelo LLM"

    async def test_process_preserves_original_payload(self, worker, mock_redis, llm_payload):
        """Arrange: a payload carrying a document id.
        Act: process it.
        Assert: the published result keeps that document id.
        """
        await worker.process(llm_payload)

        call_args = mock_redis.xadd.call_args[0]
        document_id = call_args[1].get("document_id", "")
        assert document_id == "bo-123"

    async def test_process_empty_text(self, worker, mock_redis, mock_llm):
        """Arrange: a payload with an empty text field.
        Act: process it.
        Assert: no predict call and no publish.
        """
        await worker.process({"document_id": "bo-123", "text": ""})

        mock_llm.predict.assert_not_called()
        mock_redis.xadd.assert_not_called()

    async def test_process_blank_text(self, worker, mock_redis, mock_llm):
        """Arrange: a payload whose text is only whitespace.
        Act: process it.
        Assert: no predict call and no publish.
        """
        await worker.process({"document_id": "bo-123", "text": "   "})

        mock_llm.predict.assert_not_called()
        mock_redis.xadd.assert_not_called()

    async def test_process_missing_document_id_raises(self, worker):
        """Arrange: a payload with text but no document id.
        Act: process it.
        Assert: ValueError naming document_id, so it takes the retry/DLQ path.
        """
        with pytest.raises(ValueError, match="document_id"):
            await worker.process({"text": "algum texto"})

    async def test_process_empty_summary_skips(self, worker, mock_redis, mock_llm):
        """Arrange: a model that returns an empty summary.
        Act: process a valid payload.
        Assert: nothing is published.
        """
        mock_llm.predict.return_value = [{"text": ""}]

        await worker.process({"document_id": "d1", "text": "conteúdo do OCR"})

        mock_redis.xadd.assert_not_called()

    async def test_process_blank_summary_skips(self, worker, mock_redis, mock_llm):
        """Arrange: a model that returns a whitespace-only summary.
        Act: process a valid payload.
        Assert: nothing is published.
        """
        mock_llm.predict.return_value = [{"text": "   "}]

        await worker.process({"document_id": "d1", "text": "conteúdo do OCR"})

        mock_redis.xadd.assert_not_called()

    async def test_model_loaded_lazily(self, mock_redis, mock_llm):
        """Arrange: a new worker whose model and storage are still None.
        Act: process the first payload.
        Assert: the model is built on demand and predict runs once.
        """
        with (
            patch("workers.llm_worker.LLMModel") as mock_llm_cls,
            patch("workers.llm_worker.StorageService") as mock_storage_cls,
        ):
            mock_llm_cls.return_value = mock_llm
            mock_storage_cls.return_value.save.return_value = None

            w = LlmWorker()
            w.redis = mock_redis
            assert w._llm is None

            await w.process({"document_id": "d1", "text": "conteúdo do OCR"})

            assert w._llm is mock_llm
            mock_llm.predict.assert_called_once()

    async def test_unload_on_shutdown(self, mock_llm):
        """Arrange: a worker with an injected model and run() forced to raise.
        Act: run the worker.
        Assert: the exception propagates and the model is unloaded in finally.
        """
        with (
            patch("workers.llm_worker.LLMModel") as mock_llm_cls,
            patch("workers.llm_worker.BaseWorker.run") as mock_run,
        ):
            mock_llm_cls.return_value = mock_llm
            mock_run.side_effect = Exception("force stop")

            w = LlmWorker()
            w._llm = mock_llm

            with pytest.raises(Exception, match="force stop"):
                w.run()

            mock_llm.unload.assert_called_once()

    async def test_build_prompt(self, worker):
        """Arrange: a document text.
        Act: build the summary prompt.
        Assert: it carries the text and both summary instructions.
        """
        text = "conteúdo do documento"
        prompt = worker._build_prompt(text)
        assert "conteúdo do documento" in prompt
        assert "Resumo" in prompt
        assert "Fatos principais" in prompt

    async def test_predict_many_delegates_to_llm_model(self, worker, mock_llm):
        """Arrange: a model returning two result dicts.
        Act: predict many from two prompts.
        Assert: one LLMModel.predict call with both prompts and the texts
        extracted in order.
        """
        mock_llm.predict.return_value = [{"text": "a"}, {"text": "b"}]

        result = worker._predict_many(["prompt 1", "prompt 2"])

        mock_llm.predict.assert_called_once_with(["prompt 1", "prompt 2"])
        assert result == ["a", "b"]

    async def test_batch_sends_every_prompt_in_one_predict_call(
        self, worker, mock_redis, mock_llm, llm_payload
    ):
        """Arrange: two valid payloads and a model returning two summaries.
        Act: process both in one batch.
        Assert: one predict call with two prompts and two publishes — a batch is
        one vLLM call, not one per document.
        """
        mock_llm.predict.return_value = [{"text": "resumo 1"}, {"text": "resumo 2"}]
        outros = {**llm_payload, "document_id": "bo-456", "text": "outro texto"}

        errors = await worker.process_batch([llm_payload, outros])

        assert errors == [None, None]
        mock_llm.predict.assert_called_once()
        assert len(mock_llm.predict.call_args[0][0]) == 2
        assert mock_redis.xadd.call_count == 2

    async def test_batch_fails_only_the_malformed_item(
        self, worker, mock_redis, mock_llm, llm_payload
    ):
        """Arrange: a batch with one malformed payload followed by a valid one.
        Act: process the batch.
        Assert: only the malformed item reports ValueError; the valid one still
        gets one predict and one publish.
        """
        mock_llm.predict.return_value = [{"text": "resumo"}]

        errors = await worker.process_batch([{"text": "sem document_id"}, llm_payload])

        assert isinstance(errors[0], ValueError)
        assert errors[1] is None
        assert len(mock_llm.predict.call_args[0][0]) == 1
        assert mock_redis.xadd.call_count == 1

    async def test_batch_reports_bronze_failure_for_its_item(
        self, worker, mock_redis, mock_llm, llm_payload
    ):
        """Arrange: a storage service whose save fails.
        Act: process a batch with one valid payload.
        Assert: that item reports the error and nothing is published.
        """
        worker._ensure_storage()
        worker._storage.save.side_effect = Exception("minio down")

        errors = await worker.process_batch([llm_payload])

        assert isinstance(errors[0], Exception)
        mock_redis.xadd.assert_not_awaited()

    async def test_save_failure_fails_before_publishing(self, worker, mock_redis, llm_payload):
        """Arrange: a storage service whose save fails, on the single-item path.
        Act: process one payload.
        Assert: the error propagates and nothing is published.
        """
        worker._ensure_storage()
        worker._storage.save.side_effect = Exception("minio down")

        with pytest.raises(Exception, match="minio down"):
            await worker.process(llm_payload)

        mock_redis.xadd.assert_not_awaited()
