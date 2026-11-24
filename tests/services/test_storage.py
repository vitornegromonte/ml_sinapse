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
        """Arrange: a storage service and a bronze object.
        Act: save it.
        Assert: the deterministic object path is returned and put_object runs.
        """
        path = storage.save("ocr", "doc-1", TEXT.encode())
        assert path == "ml-results/ocr/doc-1"
        mock_minio.put_object.assert_called_once()

    def test_save_failure_raises(self, storage, mock_minio):
        """Arrange: a MinIO client whose put_object fails.
        Act: save a bronze object.
        Assert: the error propagates, so the task retries or dead-letters
        instead of being acked with the text only in Redis.
        """
        mock_minio.put_object.side_effect = Exception("connection refused")

        with pytest.raises(Exception, match="connection refused"):
            storage.save("ocr", "doc-1", TEXT.encode())

    def test_load_success(self, storage, mock_minio):
        """Arrange: a MinIO response containing the saved text.
        Act: load the bronze object.
        Assert: the decoded text is returned and the exact bucket/key are read.
        """
        mock_response = MagicMock()
        mock_response.read.return_value = TEXT.encode()
        mock_minio.get_object.return_value = mock_response

        result = storage.load("ocr", "doc-1")
        assert result == TEXT
        mock_minio.get_object.assert_called_once_with(
            "bronze", "ml-results/ocr/doc-1"
        )

    def test_load_not_found_returns_none(self, storage, mock_minio):
        """Arrange: a MinIO client that reports the object is missing.
        Act: load it.
        Assert: None, so a missing object is a readable state.
        """
        mock_minio.get_object.side_effect = Exception("not found")
        result = storage.load("ocr", "doc-inexistente")
        assert result is None

    def test_load_failure_logs_warning(self, storage, mock_minio):
        """Arrange: a MinIO client whose read fails.
        Act: load the object.
        Assert: None is returned (the warning is logged by the implementation,
        not asserted here).
        """
        mock_minio.get_object.side_effect = Exception("connection refused")
        result = storage.load("ocr", "doc-1")
        assert result is None

    def test_save_and_load_roundtrip(self, storage, mock_minio):
        """Arrange: a MinIO response for an object that was just saved.
        Act: save and then load the same object.
        Assert: the text survives the round trip unchanged.
        """
        mock_response = MagicMock()
        mock_response.read.return_value = TEXT.encode()
        mock_minio.get_object.return_value = mock_response

        storage.save("ocr", "doc-roundtrip", TEXT.encode())
        result = storage.load("ocr", "doc-roundtrip")
        assert result == TEXT


class TestBuildMinioClient:
    def test_https_endpoint_is_stripped(self):
        """Arrange: a storage service configured with an https:// endpoint.
        Act: construct it with Minio patched.
        Assert: the scheme is stripped and secure=False, matching the previous
        http:// handling (B17: https:// used to survive and mangle every URL).
        """
        with patch("services.storage.Minio") as minio_cls:
            StorageService(
                endpoint="https://minio:9000",
                access_key="a",
                secret_key="b",
                bucket="bronze",
            )

        assert minio_cls.call_args.args[0] == "minio:9000"
        assert minio_cls.call_args.kwargs["secure"] is False
