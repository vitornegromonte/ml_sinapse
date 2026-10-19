from unittest.mock import AsyncMock

import pytest
from PIL import Image


@pytest.fixture
def mock_redis():
    redis = AsyncMock()
    redis.xadd = AsyncMock()
    # ResultPublisher reads XLEN after each publish to notice a stream sitting
    # at its cap; a bare AsyncMock isn't orderable against an int.
    redis.xlen = AsyncMock(return_value=0)
    return redis


@pytest.fixture
def sample_image():
    return Image.new("RGB", (100, 50), color="white")


@pytest.fixture
def ocr_payload():
    return {"document_id": "test-001", "image_url": "https://example.com/img.png"}
