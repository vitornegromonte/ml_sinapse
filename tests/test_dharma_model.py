from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from models.ocr.dharma import DharmaOcrModel


def _fake_chat_output(text: str, finish_reason: str = "stop") -> MagicMock:
    completion = MagicMock()
    completion.text = text
    completion.finish_reason = finish_reason
    request_output = MagicMock()
    request_output.outputs = [completion]
    return [request_output]


class TestDharmaOcrModel:
    def test_load_constructs_vllm_engine_with_tuning_params(self):
        """Arrange: a Dharma model with explicit tuning parameters.
        Act: load it with vLLM patched.
        Assert: the vLLM engine is built once with those parameters and kept on
        the instance.
        """
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
                limit_mm_per_prompt={"image": 1, "video": 0},
                max_model_len=4096,
            )
            assert model._llm is mock_llm_cls.return_value

    def test_load_omits_max_model_len_when_unset(self):
        """Arrange: a Dharma model with no max_model_len configured.
        Act: load it with vLLM patched.
        Assert: max_model_len is not passed, letting vLLM infer it.
        """
        with patch("vllm.LLM") as mock_llm_cls:
            model = DharmaOcrModel("fake/qwen", "/cache", 0)
            model.load()

            assert "max_model_len" not in mock_llm_cls.call_args.kwargs

    def test_predict_returns_text_from_chat(self):
        """Arrange: a loaded model whose chat returns one completion.
        Act: predict from an image.
        Assert: the completion text is returned and SamplingParams gets the
        requested max_tokens, temperature 0 and the model's repetition penalty.
        """
        model = DharmaOcrModel("fake/qwen", "/cache", 0)
        model._llm = MagicMock()
        model._llm.chat.return_value = _fake_chat_output("texto extraído")

        with patch("vllm.SamplingParams") as mock_sp:
            img = Image.new("RGB", (100, 50), color="white")
            result = model.predict(img, max_tokens=256)

            assert result == "texto extraído"
            mock_sp.assert_called_once_with(
                max_tokens=256, temperature=0, repetition_penalty=1.05
            )

    def test_predict_sends_system_and_image_url_messages(self):
        """Arrange: a loaded model and a small in-memory image.
        Act: predict from the image.
        Assert: chat receives a system message and a user message whose first
        part is the image as a base64 data URL and second is the text prompt.
        """
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

    @pytest.mark.parametrize(
        "finish_reason, warned", [("length", True), ("stop", False)]
    )
    def test_predict_warns_only_when_output_is_truncated(
        self, caplog, finish_reason, warned
    ):
        """Arrange: a loaded model whose chat output has a finish_reason.
        Act: predict an image.
        Assert: a truncation warning is logged only for finish_reason "length".
        """
        model = DharmaOcrModel("Dharma-AI/Dharma-OCR-LITE", "/cache", 0)
        model._llm = MagicMock()
        model._llm.chat.return_value = _fake_chat_output("texto", finish_reason)

        with patch("vllm.SamplingParams"), caplog.at_level("WARNING"):
            model.predict(Image.new("RGB", (10, 10)), max_tokens=64)

        assert ("truncated" in caplog.text) is warned

    def test_predict_batch_sends_one_chat_call_with_one_conversation_per_image(self):
        """Arrange: a loaded model whose chat returns 3 outputs.
        Act: predict_batch 3 images.
        Assert: chat is called once with 3 conversations, and the texts come
        back in input order.
        """
        model = DharmaOcrModel("Dharma-AI/Dharma-OCR-LITE", "/cache", 0)
        model._llm = MagicMock()
        outputs = [_fake_chat_output(t)[0] for t in ("a", " b ", "c")]
        model._llm.chat.return_value = outputs
        images = [Image.new("RGB", (10, 10)) for _ in range(3)]

        with patch("vllm.SamplingParams"):
            result = model.predict_batch(images, max_tokens=64)

        model._llm.chat.assert_called_once()
        conversations = model._llm.chat.call_args[0][0]
        assert len(conversations) == 3
        assert all(c[0]["role"] == "system" for c in conversations)
        assert result == ["a", "b", "c"]

    def test_predict_batch_uses_the_repetition_penalty(self):
        """Arrange: a loaded model.
        Act: predict_batch two images with SamplingParams watched.
        Assert: SamplingParams gets the model's repetition penalty, once for
        the whole batch.
        """
        model = _loaded_model()
        model._llm.chat.return_value = [_fake_chat_output(t)[0] for t in ("a", "b")]
        images = [Image.new("RGB", (10, 10)) for _ in range(2)]

        with patch("vllm.SamplingParams") as mock_sp:
            model.predict_batch(images, max_tokens=64)

        mock_sp.assert_called_once_with(max_tokens=64, temperature=0, repetition_penalty=1.05)

    def test_predict_batch_with_no_images_skips_the_engine(self):
        """Arrange: a loaded model.
        Act: predict_batch an empty list.
        Assert: an empty list, and chat is not called.
        """
        model = DharmaOcrModel("Dharma-AI/Dharma-OCR-LITE", "/cache", 0)
        model._llm = MagicMock()

        result = model.predict_batch([], max_tokens=64)

        assert result == []
        model._llm.chat.assert_not_called()

    def test_predict_batch_raises_on_output_count_mismatch(self):
        """Arrange: a loaded model whose chat returns fewer outputs than images.
        Act: predict_batch 2 images.
        Assert: RuntimeError, so a texts/images misalignment is never used.
        """
        model = DharmaOcrModel("Dharma-AI/Dharma-OCR-LITE", "/cache", 0)
        model._llm = MagicMock()
        model._llm.chat.return_value = _fake_chat_output("only one")
        images = [Image.new("RGB", (10, 10)) for _ in range(2)]

        with patch("vllm.SamplingParams"), pytest.raises(RuntimeError, match="1 outputs for 2"):
            model.predict_batch(images, max_tokens=64)

    def test_predict_batch_raises_if_not_loaded(self):
        """Arrange: a model that was never loaded.
        Act: predict_batch one image.
        Assert: RuntimeError asking for load().
        """
        model = DharmaOcrModel("Dharma-AI/Dharma-OCR-LITE", "/cache", 0)

        with patch("vllm.SamplingParams"), pytest.raises(RuntimeError, match="not loaded"):
            model.predict_batch([Image.new("RGB", (10, 10))], max_tokens=64)

    def test_predict_batch_warns_for_each_truncated_page(self, caplog):
        """Arrange: a batch where only the second output hit max_tokens.
        Act: predict_batch 2 images.
        Assert: exactly one truncation warning is logged.
        """
        model = DharmaOcrModel("Dharma-AI/Dharma-OCR-LITE", "/cache", 0)
        model._llm = MagicMock()
        model._llm.chat.return_value = [
            _fake_chat_output("ok", "stop")[0],
            _fake_chat_output("cut", "length")[0],
        ]
        images = [Image.new("RGB", (10, 10)) for _ in range(2)]

        with patch("vllm.SamplingParams"), caplog.at_level("WARNING"):
            model.predict_batch(images, max_tokens=64)

        assert caplog.text.count("truncated") == 1

    def test_predict_raises_if_not_loaded(self):
        """Arrange: a model that was constructed but not loaded.
        Act: predict from an image.
        Assert: RuntimeError naming the missing load, not a silent AttributeError.
        """
        model = DharmaOcrModel("fake/qwen", "/cache", 0)
        img = Image.new("RGB", (10, 10))
        with pytest.raises(RuntimeError, match="not loaded"):
            model.predict(img, max_tokens=128)

    def test_unload_clears_llm_reference(self):
        """Arrange: a loaded model on a machine with no CUDA.
        Act: unload it.
        Assert: the vLLM reference is cleared.
        """
        model = DharmaOcrModel("fake/qwen", "/cache", 0)
        model._llm = MagicMock()
        with patch("torch.cuda.is_available", return_value=False):
            model.unload()
        assert model._llm is None

    def test_device_property_cpu_fallback(self):
        """Arrange: a model configured for GPU 0 on a machine with no CUDA.
        Act: read the device.
        Assert: it falls back to "cpu".
        """
        model = DharmaOcrModel("fake/qwen", "/cache", 0)
        with patch("torch.cuda.is_available", return_value=False):
            assert model.device == "cpu"

    def test_device_property_reports_configured_gpu(self):
        """Arrange: a model configured for GPU 2 with CUDA available.
        Act: read the device.
        Assert: "cuda:2" — the configured index is kept.
        """
        model = DharmaOcrModel("fake/qwen", "/cache", 2)
        with patch("torch.cuda.is_available", return_value=True):
            assert model.device == "cuda:2"


def _loaded_model() -> DharmaOcrModel:
    model = DharmaOcrModel("Dharma-AI/Dharma-OCR-LITE", "/cache", 0)
    model._llm = MagicMock()
    return model


class TestDharmaJsonOutput:
    @pytest.mark.parametrize("raw, expected", [
        pytest.param(
            '{"header": "H", "margin": null, "footer": "F", "text": "Linha 1\\nLinha \\"2\\""}',
            'Linha 1\nLinha "2"',
            id="body-text-only-with-real-newlines",
        ),
        pytest.param('{"header": "H", "text": null}', "", id="null-text-is-empty"),
        pytest.param('{"text": ""}', "", id="empty-text"),
        pytest.param('```json\n{"text": "oi"}\n```', "oi", id="fenced-json"),
        pytest.param("texto puro sem json", "texto puro sem json", id="plain-text-unchanged"),
        pytest.param('{"foo": 1}', '{"foo": 1}', id="json-without-text-field"),
        pytest.param('{"text": "  espaço  "}', "espaço", id="text-is-stripped"),
        pytest.param('{"text": "a\\\\"}', "a\\", id="escaped-backslash-at-end"),
    ])
    def test_predict_returns_only_the_text_field(self, raw, expected):
        """Arrange: a loaded model whose chat answers with `raw`.
        Act: predict one image.
        Assert: the page's body text comes back as plain text.
        """
        model = _loaded_model()
        model._llm.chat.return_value = _fake_chat_output(raw)

        with patch("vllm.SamplingParams"):
            result = model.predict(Image.new("RGB", (10, 10)), max_tokens=64)

        assert result == expected

    @pytest.mark.parametrize("raw, expected", [
        pytest.param('{"header": null, "text": "abc\\ndef \\u00e9 ghi\\', "abc\ndef é ghi", id="cut-after-lone-backslash"),
        pytest.param('{"text": "abc \\u00e', "abc", id="cut-inside-unicode-escape"),
        pytest.param('{"header": "x", "text": "meio', "meio", id="cut-in-the-middle-of-a-word"),
    ])
    def test_predict_keeps_the_partial_text_of_cut_off_json(self, raw, expected, caplog):
        """Arrange: a chat answer cut off by max_tokens inside the text field.
        Act: predict one image.
        Assert: the partial body text is returned and a warning says it is partial.
        """
        model = _loaded_model()
        model._llm.chat.return_value = _fake_chat_output(raw, "length")

        with patch("vllm.SamplingParams"), caplog.at_level("WARNING"):
            result = model.predict(Image.new("RGB", (10, 10)), max_tokens=64)

        assert result == expected
        assert "partial" in caplog.text

    def test_predict_batch_parses_every_page(self):
        """Arrange: a batch of 3 JSON answers, one with no body text.
        Act: predict_batch 3 images.
        Assert: three plain texts in input order.
        """
        model = _loaded_model()
        model._llm.chat.return_value = [
            _fake_chat_output(raw)[0]
            for raw in ('{"text": "a"}', '{"header": "h", "text": null}', '{"text": "c\\nd"}')
        ]
        images = [Image.new("RGB", (10, 10)) for _ in range(3)]

        with patch("vllm.SamplingParams"):
            result = model.predict_batch(images, max_tokens=64)

        assert result == ["a", "", "c\nd"]

    def test_page_content_is_never_logged(self, caplog):
        """Arrange: a cut-off answer with a recognizable body text.
        Act: predict one image with log capture on.
        Assert: the text is absent from every log line.
        """
        model = _loaded_model()
        model._llm.chat.return_value = _fake_chat_output(
            '{"text": "conteudo-sensivel-do-bo', "length"
        )

        with patch("vllm.SamplingParams"), caplog.at_level("DEBUG"):
            model.predict(Image.new("RGB", (10, 10)), max_tokens=64)

        assert "conteudo-sensivel-do-bo" not in caplog.text
