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

    def test_save_failure_logs_warning(self, storage, mock_minio):
        mock_minio.put_object.side_effect = Exception("connection refused")
        path = storage.save("ocr", "doc-1", TEXT.encode())
        assert path is None

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
