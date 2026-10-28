from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from models.ocr.mlflow_model import MlflowOcrModel


@pytest.fixture
def mock_tensors():
    tensors = MagicMock()
    tensors.to.return_value = tensors
    tensors.input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    tensors.__getitem__.side_effect = lambda k: (
        torch.tensor([[1, 2, 3, 4, 5]]) if k == "input_ids"
        else torch.tensor([[1, 1, 1, 1, 1]])
    )
    return tensors


def _make_mlflow_loaded(model_cls_name: str, mock_tensors):
    """Cria um objeto fake que simula o retorno de mlflow.transformers.load_model."""
    model = MagicMock()
    model.__class__.__name__ = model_cls_name
    model.device = "cpu"
    model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 10, 20, 30]])

    processor = MagicMock()
    processor.apply_chat_template.return_value = "template"
    processor.return_value = mock_tensors
    processor.batch_decode.return_value = ["mlflow result"]
    processor.decode.return_value = "glm mlflow result"

    loaded = MagicMock()
    loaded.model = model
    loaded.processor = processor
    return loaded


class TestMlflowOcrModel:
    def test_architecture_qwen(self, mock_tensors):
        """Arrange: an MLflow model whose class is Qwen.
        Act: load it and ask for its architecture.
        Assert: "qwen".
        """
        loaded = _make_mlflow_loaded("Qwen2_5_VLForConditionalGeneration", mock_tensors)

        with patch("mlflow.transformers.load_model", return_value=loaded):
            model = MlflowOcrModel("models:/ocr-model/latest", gpu_device=0)
            model.load()
            assert model._model_architecture() == "qwen"

    def test_architecture_glm(self, mock_tensors):
        """Arrange: an MLflow model whose class is GLM.
        Act: load it and ask for its architecture.
        Assert: "glm".
        """
        loaded = _make_mlflow_loaded("ChatGLMModel", mock_tensors)

        with patch("mlflow.transformers.load_model", return_value=loaded):
            model = MlflowOcrModel("models:/ocr-model/latest", gpu_device=0)
            model.load()
            assert model._model_architecture() == "glm"

    def test_architecture_defaults_to_qwen(self, mock_tensors):
        """Arrange: an MLflow model with an unrecognized class name.
        Act: load it and ask for its architecture.
        Assert: it falls back to "qwen".
        """
        loaded = _make_mlflow_loaded("UnknownArchitecture", mock_tensors)

        with patch("mlflow.transformers.load_model", return_value=loaded):
            model = MlflowOcrModel("models:/ocr-model/latest", gpu_device=0)
            model.load()
            assert model._model_architecture() == "qwen"

    def test_predict_qwen_returns_string(self, mock_tensors):
        """Arrange: a loaded Qwen MLflow model with vision processing patched.
        Act: predict from an image.
        Assert: a string with the decoded model output.
        """
        loaded = _make_mlflow_loaded("Qwen2_5_VLForConditionalGeneration", mock_tensors)

        with (
            patch("mlflow.transformers.load_model", return_value=loaded),
            patch("qwen_vl_utils.process_vision_info") as mock_pvi,
        ):
            mock_pvi.return_value = ([Image.new("RGB", (10, 10))], None)

            model = MlflowOcrModel("models:/ocr-model/latest", gpu_device=0)
            model.load()

            img = Image.new("RGB", (100, 50), color="white")
            result = model.predict(img, max_tokens=128)

            assert isinstance(result, str)
            assert result == "mlflow result"

    def test_predict_glm_returns_string(self, mock_tensors):
        """Arrange: a loaded GLM MLflow model.
        Act: predict from an image.
        Assert: a string is returned.
        """
        loaded = _make_mlflow_loaded("ChatGLMModel", mock_tensors)

        with patch("mlflow.transformers.load_model", return_value=loaded):
            model = MlflowOcrModel("models:/ocr-model/latest", gpu_device=0)
            model.load()

            img = Image.new("RGB", (100, 50), color="white")
            result = model.predict(img, max_tokens=128)

            assert isinstance(result, str)

    def test_predict_qwen_prompt_override(self, mock_tensors):
        """Arrange: a loaded Qwen model and a custom prompt.
        Act: predict with prompt="PROMPT DO CHAMADOR".
        Assert: the custom prompt replaces USER_PROMPT in the chat template —
        the AbstractOcrModel contract cannot differ per backend (this used to
        raise TypeError).
        """
        loaded = _make_mlflow_loaded("Qwen2_5_VLForConditionalGeneration", mock_tensors)

        with (
            patch("mlflow.transformers.load_model", return_value=loaded),
            patch("qwen_vl_utils.process_vision_info") as mock_pvi,
        ):
            mock_pvi.return_value = ([Image.new("RGB", (10, 10))], None)
            model = MlflowOcrModel("models:/ocr-model/latest", gpu_device=0)
            model.load()

            model.predict(
                Image.new("RGB", (10, 10)), max_tokens=8, prompt="PROMPT DO CHAMADOR"
            )

            messages = model._processor.apply_chat_template.call_args[0][0]
            assert messages[1]["content"][1]["text"] == "PROMPT DO CHAMADOR"

    def test_predict_glm_prompt_override(self, mock_tensors):
        """Arrange: a loaded GLM model and a custom prompt.
        Act: predict with prompt="PROMPT DO CHAMADOR".
        Assert: the custom prompt is used in the GLM message.
        """
        loaded = _make_mlflow_loaded("ChatGLMModel", mock_tensors)

        with patch("mlflow.transformers.load_model", return_value=loaded):
            model = MlflowOcrModel("models:/ocr-model/latest", gpu_device=0)
            model.load()

            model.predict(
                Image.new("RGB", (10, 10)), max_tokens=8, prompt="PROMPT DO CHAMADOR"
            )

            messages = model._processor.apply_chat_template.call_args[0][0]
            assert messages[0]["content"][1]["text"] == "PROMPT DO CHAMADOR"

    def test_device_property_reports_configured_gpu(self):
        """Arrange: an unloaded model configured for GPU 1, CUDA available.
        Act: read the device.
        Assert: "cuda:1" from the configured gpu_device.
        """
        model = MlflowOcrModel("models:/ocr-model/latest", gpu_device=1)
        model._model = None
        with patch("torch.cuda.is_available", return_value=True):
            assert model.device == "cuda:1"

    def test_device_property_cpu_fallback(self):
        """Arrange: an unloaded model configured for GPU 1, CUDA unavailable.
        Act: read the device.
        Assert: it falls back to "cpu".
        """
        model = MlflowOcrModel("models:/ocr-model/latest", gpu_device=1)
        model._model = None
        with patch("torch.cuda.is_available", return_value=False):
            assert model.device == "cpu"

    def test_unload_clears_model(self, mock_tensors):
        """Arrange: a loaded Qwen MLflow model with model and processor set.
        Act: unload it.
        Assert: both references are cleared.
        """
        loaded = _make_mlflow_loaded("Qwen2_5_VLForConditionalGeneration", mock_tensors)

        with patch("mlflow.transformers.load_model", return_value=loaded):
            model = MlflowOcrModel("models:/ocr-model/latest", gpu_device=0)
            model.load()
            assert model._model is not None
            assert model._processor is not None

            model.unload()
            assert model._model is None
            assert model._processor is None
