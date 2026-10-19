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
        """process deve montar prompt, chamar predict e publicar resultado."""
        await worker.process(llm_payload)

        mock_llm.predict.assert_called_once()
        args, _ = mock_llm.predict.call_args
        prompt = args[0]
        assert "texto extraído via OCR" in prompt
        assert "Resumo" in prompt

        mock_redis.xadd.assert_awaited_once()
        call_args = mock_redis.xadd.call_args[0]
        assert "llm:tasks" in call_args[0]
        assert call_args[1]["document_id"] == "bo-123"
        assert call_args[1]["text"] == "resumo gerado pelo LLM"

    async def test_process_preserves_original_payload(self, worker, mock_redis, llm_payload):
        """result_data deve preservar campos extras do payload original."""
        await worker.process(llm_payload)

        call_args = mock_redis.xadd.call_args[0]
        document_id = call_args[1].get("document_id", "")
        assert document_id == "bo-123"

    async def test_process_empty_text(self, worker, mock_redis, mock_llm):
        """process com texto vazio não deve chamar predict nem publicar."""
        await worker.process({"document_id": "bo-123", "text": ""})

        mock_llm.predict.assert_not_called()
        mock_redis.xadd.assert_not_called()

    async def test_process_blank_text(self, worker, mock_redis, mock_llm):
        """process com texto de espaços não deve chamar predict nem publicar."""
        await worker.process({"document_id": "bo-123", "text": "   "})

        mock_llm.predict.assert_not_called()
        mock_redis.xadd.assert_not_called()

    async def test_process_missing_document_id_raises(self, worker):
        """process sem document_id deve levantar ValueError."""
        with pytest.raises(ValueError, match="document_id"):
            await worker.process({"text": "algum texto"})

    async def test_process_empty_summary_skips(self, worker, mock_redis, mock_llm):
        """summary vazio do LLM não deve publicar resultado."""
        mock_llm.predict.return_value = [{"text": ""}]

        await worker.process({"document_id": "d1", "text": "conteúdo do OCR"})

        mock_redis.xadd.assert_not_called()

    async def test_process_blank_summary_skips(self, worker, mock_redis, mock_llm):
        """summary com só espaços do LLM não deve publicar resultado."""
        mock_llm.predict.return_value = [{"text": "   "}]

        await worker.process({"document_id": "d1", "text": "conteúdo do OCR"})

        mock_redis.xadd.assert_not_called()

    async def test_model_loaded_lazily(self, mock_redis, mock_llm):
        """process deve carregar modelo automaticamente se None."""
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
        """run deve chamar unload do modelo no finally."""
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
        """_build_prompt deve formatar o template com o texto."""
        text = "conteúdo do documento"
        prompt = worker._build_prompt(text)
        assert "conteúdo do documento" in prompt
        assert "Resumo" in prompt
        assert "Fatos principais" in prompt

    async def test_predict_delegates_to_llm_model(self, worker, mock_llm):
        """_predict deve delegar ao LLMModel.predict e extrair texto."""
        result = worker._predict("meu prompt")
        mock_llm.predict.assert_called_once_with("meu prompt")
        assert result == "resumo gerado pelo LLM"

    async def test_save_failure_fails_before_publishing(self, worker, mock_redis, llm_payload):
        """MinIO fora do ar: a task falha (retry/DLQ) e nada é publicado."""
        worker._ensure_storage()
        worker._storage.save.side_effect = Exception("minio down")

        with pytest.raises(Exception, match="minio down"):
            await worker.process(llm_payload)

        mock_redis.xadd.assert_not_awaited()
