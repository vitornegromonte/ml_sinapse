import sys
import types
from importlib.util import find_spec
from unittest.mock import AsyncMock

import pytest
from PIL import Image

# vLLM ships CUDA-only wheels, so the CI environment (requirements-test.txt)
# does not install it. The tests that use it patch vllm.LLM /
# vllm.SamplingParams, so they only need the module to exist to patch. Stand in
# only when it is genuinely absent — the GPU image runs them against the real
# package. Limitation: a real vLLM API rename would not be caught here.
if find_spec("vllm") is None:
    _vllm = types.ModuleType("vllm")
    _vllm.LLM = None  # replaced by the tests that patch it
    _vllm.SamplingParams = None
    sys.modules["vllm"] = _vllm


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
