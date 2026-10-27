from unittest.mock import MagicMock, patch

import pytest

from services.storage import StorageService


@pytest.fixture
def mock_minio():
    with patch("services.storage.Minio") as m:
        client = MagicMock()
        m.return_value = client
        yield client


@pytest.fixture
def storage(mock_minio):
    return StorageService(
        endpoint="http://minio:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        bucket="bronze",
    )


TEXT = "conteudo do OCR"


class TestStorageService:
    def test_save_success(self, storage, mock_minio):
        path = storage.save("ocr", "doc-1", TEXT.encode())
        assert path == "ml-results/ocr/doc-1"
        mock_minio.put_object.assert_called_once()

    def test_save_failure_raises(self, storage, mock_minio):
        """Falha de escrita no bronze deve propagar, não retornar None: a task
        precisa cair no retry/DLQ em vez de ser acked com o texto só no Redis."""
        mock_minio.put_object.side_effect = Exception("connection refused")

        with pytest.raises(Exception, match="connection refused"):
            storage.save("ocr", "doc-1", TEXT.encode())

    def test_load_success(self, storage, mock_minio):
        mock_response = MagicMock()
        mock_response.read.return_value = TEXT.encode()
        mock_minio.get_object.return_value = mock_response

        result = storage.load("ocr", "doc-1")
        assert result == TEXT
        mock_minio.get_object.assert_called_once_with(
            "bronze", "ml-results/ocr/doc-1"
        )

    def test_load_not_found_returns_none(self, storage, mock_minio):
        mock_minio.get_object.side_effect = Exception("not found")
        result = storage.load("ocr", "doc-inexistente")
        assert result is None

    def test_load_failure_logs_warning(self, storage, mock_minio):
        mock_minio.get_object.side_effect = Exception("connection refused")
        result = storage.load("ocr", "doc-1")
        assert result is None

    def test_save_and_load_roundtrip(self, storage, mock_minio):
        mock_response = MagicMock()
        mock_response.read.return_value = TEXT.encode()
        mock_minio.get_object.return_value = mock_response

        storage.save("ocr", "doc-roundtrip", TEXT.encode())
        result = storage.load("ocr", "doc-roundtrip")
        assert result == TEXT


class TestBuildMinioClient:
    def test_https_endpoint_is_stripped(self):
        """B17: storage.py stripped only http://, so an https:// endpoint
        survived into the client and every request URL was mangled."""
        with patch("services.storage.Minio") as minio_cls:
            StorageService(
                endpoint="https://minio:9000",
                access_key="a",
                secret_key="b",
                bucket="bronze",
            )

        assert minio_cls.call_args.args[0] == "minio:9000"
        assert minio_cls.call_args.kwargs["secure"] is False
