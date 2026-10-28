from unittest.mock import MagicMock, patch

import pytest

from config import Settings
from workers.ner_worker import NerWorker


def _fake_entity(text: str, label: str, start: int, end: int) -> MagicMock:
    ent = MagicMock()
    ent.text = text
    ent.label_ = label
    ent.start_char = start
    ent.end_char = end
    return ent


@pytest.fixture
def mock_nlp():
    """Fake spaCy Language: nlp(text) -> doc with .ents (11 domain labels, see ADR-005)."""
    def _call(text):
        doc = MagicMock()
        doc.ents = [_fake_entity("João da Silva", "PESSOA", 19, 32)]
        return doc
    return MagicMock(side_effect=_call)


@pytest.fixture
def worker(mock_redis, mock_nlp):
    """NerWorker with Redis mocked and the spaCy model pre-injected (already loaded)."""
    w = NerWorker()
    w.redis = mock_redis
    w._nlp = mock_nlp
    yield w


@pytest.fixture
def ner_payload():
    return {"document_id": "bo-123", "text": "Boletim envolvendo João da Silva, placa ABC-1234."}


class TestNerWorker:
    async def test_process_runs_nlp_on_text(self, worker, mock_nlp, ner_payload):
        """Arrange: a NER worker with an injected spaCy pipeline.
        Act: process a payload.
        Assert: the pipeline is called once with the payload text.
        """
        await worker.process(ner_payload)
        mock_nlp.assert_called_once_with(ner_payload["text"])

    async def test_process_extracts_entity_fields(self, worker, mock_nlp, ner_payload, caplog):
        """Arrange: a pipeline returning one PESSOA entity.
        Act: process a payload with info logging captured.
        Assert: the entity text and label reach the log.
        """
        import logging
        with caplog.at_level(logging.INFO):
            await worker.process(ner_payload)

        assert "João da Silva" in caplog.text
        assert "PESSOA" in caplog.text

    async def test_process_does_not_touch_redis(self, worker, mock_redis, ner_payload):
        """Arrange: a worker with a mocked Redis client.
        Act: process a payload.
        Assert: nothing is published — this worker only logs (mock NER, ADR-005).
        """
        await worker.process(ner_payload)
        mock_redis.xadd.assert_not_called()

    async def test_model_loaded_lazily_from_local_path(self, mock_redis, ner_payload):
        """Arrange: no ner_model_uri, with spaCy and MinIO sync patched.
        Act: process the first payload.
        Assert: the model is synced from its configured bucket/prefix, then
        loaded with spacy.load from the local path.
        """
        with (
            patch("workers.ner_worker.spacy") as mock_spacy,
            patch("workers.ner_worker.sync_if_missing") as mock_sync,
        ):
            mock_spacy.load.return_value = MagicMock(side_effect=lambda t: MagicMock(ents=[]))

            w = NerWorker()
            w.redis = mock_redis
            assert w._nlp is None

            await w.process(ner_payload)

            mock_sync.assert_called_once_with(
                local_dir=w.cfg.ner_model_path,
                bucket=w.cfg.minio_bucket_models,
                prefix=w.cfg.ner_model_minio_prefix,
                endpoint=w.cfg.minio_endpoint,
                access_key=w.cfg.minio_access_key,
                secret_key=w.cfg.minio_secret_key,
            )
            mock_spacy.load.assert_called_once_with(w.cfg.ner_model_path)
            assert w._nlp is not None

    async def test_model_load_skips_minio_sync_when_prefix_unset(self, mock_redis, ner_payload):
        """Arrange: config with an empty ner_model_minio_prefix.
        Act: process the first payload.
        Assert: MinIO sync is skipped and spacy.load uses the local path.
        """
        cfg = Settings(ner_model_minio_prefix="")
        with (
            patch("workers.ner_worker.spacy") as mock_spacy,
            patch("workers.ner_worker.sync_if_missing") as mock_sync,
        ):
            mock_spacy.load.return_value = MagicMock(side_effect=lambda t: MagicMock(ents=[]))

            w = NerWorker(cfg=cfg)
            w.redis = mock_redis

            await w.process(ner_payload)

            mock_sync.assert_not_called()
            mock_spacy.load.assert_called_once_with(cfg.ner_model_path)

    async def test_model_loaded_lazily_from_mlflow_when_uri_set(self, mock_redis, ner_payload):
        """Arrange: config with an MLflow model URI, load_model patched.
        Act: process the first payload.
        Assert: the model is loaded from that URI instead of MinIO.
        """
        cfg = Settings(ner_model_uri="models:/ner-model/1")
        with patch("mlflow.spacy.load_model") as mock_load:
            mock_load.return_value = MagicMock(side_effect=lambda t: MagicMock(ents=[]))

            w = NerWorker(cfg=cfg)
            w.redis = mock_redis

            await w.process(ner_payload)

            mock_load.assert_called_once_with("models:/ner-model/1")
