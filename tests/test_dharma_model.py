from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from models.ocr.dharma import DharmaOcrModel


def _fake_chat_output(text: str) -> MagicMock:
    completion = MagicMock()
    completion.text = text
    request_output = MagicMock()
    request_output.outputs = [completion]
    return [request_output]


class TestDharmaOcrModel:
    def test_load_constructs_vllm_engine_with_tuning_params(self):
        """load deve repassar dtype/gpu_memory_utilization/max_model_len ao vLLM."""
        with patch("vllm.LLM") as mock_llm_cls:
            model = DharmaOcrModel(
                "Dharma-AI/Dharma-OCR-LITE", "/cache", 0,
                dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=4096,
            )
            model.load()

            mock_llm_cls.assert_called_once_with(
                model="Dharma-AI/Dharma-OCR-LITE",
                trust_remote_code=True,
                dtype="bfloat16",
                gpu_memory_utilization=0.85,
                download_dir="/cache",
                limit_mm_per_prompt={"image": 1},
                max_model_len=4096,
            )
            assert model._llm is mock_llm_cls.return_value

    def test_load_omits_max_model_len_when_unset(self):
        """max_model_len=None (default) não deve virar kwarg — deixa o vLLM inferir."""
        with patch("vllm.LLM") as mock_llm_cls:
            model = DharmaOcrModel("fake/qwen", "/cache", 0)
            model.load()

            assert "max_model_len" not in mock_llm_cls.call_args.kwargs

    def test_predict_returns_text_from_chat(self):
        """predict deve retornar o texto do primeiro output de LLM.chat()."""
        model = DharmaOcrModel("fake/qwen", "/cache", 0)
        model._llm = MagicMock()
        model._llm.chat.return_value = _fake_chat_output("texto extraído")

        with patch("vllm.SamplingParams") as mock_sp:
            img = Image.new("RGB", (100, 50), color="white")
            result = model.predict(img, max_tokens=256)

            assert result == "texto extraído"
            mock_sp.assert_called_once_with(max_tokens=256, temperature=0)

    def test_predict_sends_system_and_image_url_messages(self):
        """predict deve montar messages com system prompt + image_url em base64."""
        model = DharmaOcrModel("fake/qwen", "/cache", 0)
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

    def test_predict_raises_if_not_loaded(self):
        """predict antes de load() deve levantar RuntimeError, não AttributeError silencioso."""
        model = DharmaOcrModel("fake/qwen", "/cache", 0)
        img = Image.new("RGB", (10, 10))
        with pytest.raises(RuntimeError, match="not loaded"):
            model.predict(img, max_tokens=128)

    def test_unload_clears_llm_reference(self):
        model = DharmaOcrModel("fake/qwen", "/cache", 0)
        model._llm = MagicMock()
        with patch("torch.cuda.is_available", return_value=False):
            model.unload()
        assert model._llm is None

    def test_device_property_cpu_fallback(self):
        model = DharmaOcrModel("fake/qwen", "/cache", 0)
        with patch("torch.cuda.is_available", return_value=False):
            assert model.device == "cpu"

    def test_device_property_reports_configured_gpu(self):
        model = DharmaOcrModel("fake/qwen", "/cache", 2)
        with patch("torch.cuda.is_available", return_value=True):
            assert model.device == "cuda:2"
