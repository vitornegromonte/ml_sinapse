from unittest.mock import MagicMock, patch

import pytest

from models.llm_model import LLMModel


class TestLLMModelVllmOnly:
    def test_load_constructs_vllm_with_tuning_params(self):
        """load repassa os parâmetros de tuning ao vLLM."""
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
        with patch("vllm.LLM") as mock_vllm, patch("transformers.AutoTokenizer"):
            model = LLMModel("fake/model")

            model.load()

            assert "max_model_len" not in mock_vllm.call_args.kwargs

    def test_constructor_no_longer_accepts_engine(self):
        """O dispatch de engine foi removido: vLLM é o único caminho."""
        with pytest.raises(TypeError):
            LLMModel("fake/model", engine="transformers")

    def test_predict_returns_text_and_tokens(self):
        model = LLMModel("fake/model")
        model._llm = MagicMock()
        output = MagicMock()
        output.outputs = [MagicMock(text="resumo", token_ids=[1, 2, 3])]
        model._llm.generate.return_value = [output]

        with patch("vllm.SamplingParams"):
            result = model.predict("um prompt")

        assert result == [{"text": "resumo", "tokens": [1, 2, 3]}]
