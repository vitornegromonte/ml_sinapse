from unittest.mock import MagicMock, patch

import pytest

from models.llm_model import LLMModel


class TestLLMModelVllmOnly:
    def test_load_constructs_vllm_with_tuning_params(self):
        """Arrange: an LLM model with explicit tuning parameters.
        Act: load it with vLLM patched.
        Assert: vLLM is constructed once with those parameters and kept on the
        instance.
        """
        with patch("vllm.LLM") as mock_vllm, patch("transformers.AutoTokenizer"):
            model = LLMModel(
                "fake/model",
                dtype="bfloat16",
                gpu_memory_utilization=0.85,
                max_model_len=4096,
                tensor_parallel_size=2,
            )
            model.load()

            mock_vllm.assert_called_once_with(
                model="fake/model",
                dtype="bfloat16",
                gpu_memory_utilization=0.85,
                tensor_parallel_size=2,
                max_model_len=4096,
            )
            assert model._llm is mock_vllm.return_value

    def test_load_omits_max_model_len_when_unset(self):
        """Arrange: an LLM model with no max_model_len configured.
        Act: load it with vLLM patched.
        Assert: max_model_len is not passed, so vLLM keeps its native context.
        """
        with patch("vllm.LLM") as mock_vllm, patch("transformers.AutoTokenizer"):
            model = LLMModel("fake/model")

            model.load()

            assert "max_model_len" not in mock_vllm.call_args.kwargs

    def test_constructor_no_longer_accepts_engine(self):
        """Arrange: constructor arguments that still name an old engine.
        Act: build the LLM model.
        Assert: TypeError — vLLM is the only path now.
        """
        with pytest.raises(TypeError):
            LLMModel("fake/model", engine="transformers")

    def test_predict_returns_text_and_tokens(self):
        """Arrange: a loaded model whose vLLM generate returns one output.
        Act: predict from a prompt.
        Assert: the text is mapped into the worker's result shape with tokens.
        """
        model = LLMModel("fake/model")
        model._llm = MagicMock()
        output = MagicMock()
        output.outputs = [MagicMock(text="resumo", token_ids=[1, 2, 3])]
        model._llm.generate.return_value = [output]

        with patch("vllm.SamplingParams"):
            result = model.predict("um prompt")

        assert result == [{"text": "resumo", "tokens": [1, 2, 3]}]


class TestLLMModelModelSpecificSettings:
    def test_engine_kwargs_are_added_to_the_vllm_call(self):
        """Arrange: an LLM model with an extra engine argument.
        Act: load it with vLLM patched.
        Assert: the argument reaches vLLM next to the usual ones.
        """
        with patch("vllm.LLM") as mock_vllm, patch("transformers.AutoTokenizer"):
            model = LLMModel("fake/model", engine_kwargs={"language_model_only": True})

            model.load()

            assert mock_vllm.call_args.kwargs["language_model_only"] is True
            assert mock_vllm.call_args.kwargs["model"] == "fake/model"

    def test_no_engine_kwargs_adds_nothing(self):
        """Arrange: an LLM model with no engine kwargs.
        Act: load it with vLLM patched.
        Assert: only the standard arguments are passed.
        """
        with patch("vllm.LLM") as mock_vllm, patch("transformers.AutoTokenizer"):
            LLMModel("fake/model").load()

            assert set(mock_vllm.call_args.kwargs) == {
                "model", "dtype", "gpu_memory_utilization", "tensor_parallel_size"
            }

    def test_chat_template_kwargs_reach_the_template(self):
        """Arrange: a loaded model with enable_thinking=False and a fake tokenizer.
        Act: predict from a prompt.
        Assert: apply_chat_template gets enable_thinking=False, so a reasoning
        model answers directly.
        """
        model = LLMModel(
            "fake/model",
            sampling_params={"max_tokens": 64},
            chat_template_kwargs={"enable_thinking": False},
        )
        model._llm = MagicMock()
        model._tokenizer = MagicMock()
        model._tokenizer.apply_chat_template.return_value = "rendered"
        output = MagicMock()
        output.outputs = [MagicMock(text="resumo", token_ids=[1])]
        model._llm.generate.return_value = [output]

        with patch("vllm.SamplingParams"):
            model.predict("um prompt")

        _, kwargs = model._tokenizer.apply_chat_template.call_args
        assert kwargs["enable_thinking"] is False
        assert kwargs["add_generation_prompt"] is True
        assert kwargs["tokenize"] is False

    def test_no_chat_template_kwargs_keeps_the_old_call(self):
        """Arrange: a loaded model with no chat template kwargs.
        Act: predict from a prompt.
        Assert: apply_chat_template gets only the two standard arguments.
        """
        model = LLMModel("fake/model", sampling_params={"max_tokens": 64})
        model._llm = MagicMock()
        model._tokenizer = MagicMock()
        output = MagicMock()
        output.outputs = [MagicMock(text="resumo", token_ids=[1])]
        model._llm.generate.return_value = [output]

        with patch("vllm.SamplingParams"):
            model.predict("um prompt")

        _, kwargs = model._tokenizer.apply_chat_template.call_args
        assert set(kwargs) == {"tokenize", "add_generation_prompt"}

    def test_sampling_params_reach_vllm(self):
        """Arrange: a loaded model with sampling params, and a per-call override.
        Act: predict with an extra keyword.
        Assert: SamplingParams gets the configured values with the override on top.
        """
        model = LLMModel("fake/model", sampling_params={"max_tokens": 2048, "temperature": 0.7})
        model._llm = MagicMock()
        output = MagicMock()
        output.outputs = [MagicMock(text="x", token_ids=[1])]
        model._llm.generate.return_value = [output]

        with patch("vllm.SamplingParams") as mock_sp:
            model.predict("p", temperature=0.1)

        mock_sp.assert_called_once_with(max_tokens=2048, temperature=0.1)

    def test_missing_max_tokens_is_warned_about(self, caplog):
        """Arrange: sampling params without max_tokens.
        Act: build the model.
        Assert: a warning says vLLM will stop after 16 tokens.
        """
        with caplog.at_level("WARNING"):
            LLMModel("fake/model", sampling_params={"temperature": 0.7})

        assert "16 tokens" in caplog.text

    def test_max_tokens_present_gives_no_warning(self, caplog):
        """Arrange: sampling params with max_tokens.
        Act: build the model.
        Assert: no warning.
        """
        with caplog.at_level("WARNING"):
            LLMModel("fake/model", sampling_params={"max_tokens": 512})

        assert "max_tokens" not in caplog.text


class TestLLMModelRevisionPin:
    def test_load_pins_weights_and_tokenizer(self):
        """Arrange: an LLM model with a pinned revision.
        Act: load it with vLLM and the tokenizer patched.
        Assert: vLLM gets the revision for weights and tokenizer, and the
        Hugging Face tokenizer is loaded at the same commit.
        """
        with patch("vllm.LLM") as mock_vllm, patch("transformers.AutoTokenizer") as mock_tok:
            LLMModel("org/model", revision="d" * 40).load()

        assert mock_vllm.call_args.kwargs["revision"] == "d" * 40
        assert mock_vllm.call_args.kwargs["tokenizer_revision"] == "d" * 40
        mock_tok.from_pretrained.assert_called_once_with("org/model", revision="d" * 40)

    def test_load_without_a_revision_is_unchanged(self):
        """Arrange: an LLM model with no revision.
        Act: load it with vLLM and the tokenizer patched.
        Assert: no pin arguments are passed, and the tokenizer call has none.
        """
        with patch("vllm.LLM") as mock_vllm, patch("transformers.AutoTokenizer") as mock_tok:
            LLMModel("org/model").load()

        assert "revision" not in mock_vllm.call_args.kwargs
        mock_tok.from_pretrained.assert_called_once_with("org/model")
