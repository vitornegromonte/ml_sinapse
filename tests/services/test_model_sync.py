"""B17: model_sync kept its own copy of the endpoint normalisation."""
from unittest.mock import patch

import pytest

from services.model_sync import sync_if_missing


def test_sync_uses_the_shared_minio_client(tmp_path):
    """Arrange: an empty model bucket and a missing local model directory.
    Act: sync the model through the shared MinIO client builder.
    Assert: it raises FileNotFoundError and builds the client once with the
    endpoint and credentials it was given.
    """
    with patch("services.model_sync.build_minio_client") as build:
        build.return_value.list_objects.return_value = []

        with pytest.raises(FileNotFoundError):
            sync_if_missing(
                str(tmp_path / "model"),
                "bronze",
                "ocr",
                "https://minio:9000",
                "a",
                "b",
            )

    build.assert_called_once_with("https://minio:9000", "a", "b")
