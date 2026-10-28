from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from models.ocr.glm import GlmOcrModel


def _fake_chat_output(text: str) -> MagicMock:
    completion = MagicMock()
    completion.text = text
    request_output = MagicMock()
    request_output.outputs = [completion]
    return [request_output]


class TestGlmOcrModel:
    def test_load_constructs_vllm_engine_with_tuning_params(self):
        """Arrange: a GLM model with explicit tuning parameters.
        Act: load it with vLLM patched.
        Assert: the vLLM engine is built once with those parameters and kept on
        the instance.
        """
        with patch("vllm.LLM") as mock_llm_cls:
            model = GlmOcrModel(
                "zai-org/GLM-OCR", "/cache", 0,
                dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=4096,
            )
            model.load()

            mock_llm_cls.assert_called_once_with(
                model="zai-org/GLM-OCR",
                trust_remote_code=True,
                dtype="bfloat16",
                gpu_memory_utilization=0.85,
                download_dir="/cache",
                limit_mm_per_prompt={"image": 1},
                max_model_len=4096,
            )
            assert model._llm is mock_llm_cls.return_value

    def test_load_omits_max_model_len_when_unset(self):
        """Arrange: a GLM model with no max_model_len configured.
        Act: load it with vLLM patched.
        Assert: max_model_len is not passed, letting vLLM infer it.
        """
        with patch("vllm.LLM") as mock_llm_cls:
            model = GlmOcrModel("fake/glm", "/cache", 0)
            model.load()

            assert "max_model_len" not in mock_llm_cls.call_args.kwargs

    def test_predict_returns_text_from_chat(self):
        """Arrange: a loaded model whose chat returns one completion.
        Act: predict from an image.
        Assert: the completion text is returned and SamplingParams gets the
        requested max_tokens with temperature 0.
        """
        model = GlmOcrModel("fake/glm", "/cache", 0)
        model._llm = MagicMock()
        model._llm.chat.return_value = _fake_chat_output("texto extraído do glm")

        with patch("vllm.SamplingParams") as mock_sp:
            img = Image.new("RGB", (100, 50), color="white")
            result = model.predict(img, max_tokens=256)

            assert result == "texto extraído do glm"
            mock_sp.assert_called_once_with(max_tokens=256, temperature=0)

    def test_predict_sends_system_and_image_url_messages(self):
        """Arrange: a loaded model and a small in-memory image.
        Act: predict from the image.
        Assert: chat receives a system message and a user message with the
        image as a base64 data URL followed by the GLM default prompt.
        """
        model = GlmOcrModel("fake/glm", "/cache", 0)
        model._llm = MagicMock()
        model._llm.chat.return_value = _fake_chat_output("ok")

        with patch("vllm.SamplingParams"):
            img = Image.new("RGB", (10, 10), color="white")
            model.predict(img, max_tokens=128)

        messages = model._llm.chat.call_args[0][0]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        content = messages[1]["content"]
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert content[1]["type"] == "text"
        assert content[1]["text"] == GlmOcrModel.USER_PROMPT

    def test_predict_prompt_override(self):
        """Arrange: a loaded model and a custom prompt.
        Act: predict with prompt="Free OCR.".
        Assert: the custom prompt replaces the default USER_PROMPT in the
        message sent to chat.
        """
        model = GlmOcrModel("fake/glm", "/cache", 0)
        model._llm = MagicMock()
        model._llm.chat.return_value = _fake_chat_output("ok")

        with patch("vllm.SamplingParams"):
            img = Image.new("RGB", (10, 10), color="white")
            model.predict(img, max_tokens=128, prompt="Free OCR.")

        messages = model._llm.chat.call_args[0][0]
        assert messages[1]["content"][1]["text"] == "Free OCR."

    def test_predict_raises_if_not_loaded(self):
        """Arrange: a model that was constructed but not loaded.
        Act: predict from an image.
        Assert: RuntimeError naming the missing load, not a silent AttributeError.
        """
        model = GlmOcrModel("fake/glm", "/cache", 0)
        img = Image.new("RGB", (10, 10))
        with pytest.raises(RuntimeError, match="not loaded"):
            model.predict(img, max_tokens=128)

    def test_unload_clears_llm_reference(self):
        """Arrange: a loaded model on a machine with no CUDA.
        Act: unload it.
        Assert: the vLLM reference is cleared.
        """
        model = GlmOcrModel("fake/glm", "/cache", 0)
        model._llm = MagicMock()
        with patch("torch.cuda.is_available", return_value=False):
            model.unload()
        assert model._llm is None

    def test_device_property_cpu_fallback(self):
        """Arrange: a model configured for GPU 0 on a machine with no CUDA.
        Act: read the device.
        Assert: it falls back to "cpu".
        """
        model = GlmOcrModel("fake/glm", "/cache", 0)
        with patch("torch.cuda.is_available", return_value=False):
            assert model.device == "cpu"

    def test_device_property_reports_configured_gpu(self):
        """Arrange: a model configured for GPU 2 with CUDA available.
        Act: read the device.
        Assert: "cuda:2" — the configured index is kept.
        """
        model = GlmOcrModel("fake/glm", "/cache", 2)
        with patch("torch.cuda.is_available", return_value=True):
            assert model.device == "cuda:2"
