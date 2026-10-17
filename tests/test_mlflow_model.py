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
        """Modelo Qwen deve ser detectado como arquitetura 'qwen'."""
        loaded = _make_mlflow_loaded("Qwen2_5_VLForConditionalGeneration", mock_tensors)

        with patch("mlflow.transformers.load_model", return_value=loaded):
            model = MlflowOcrModel("models:/ocr-model/latest", gpu_device=0)
            model.load()
            assert model._model_architecture() == "qwen"

    def test_architecture_glm(self, mock_tensors):
        """Modelo GLM deve ser detectado como arquitetura 'glm'."""
        loaded = _make_mlflow_loaded("ChatGLMModel", mock_tensors)

        with patch("mlflow.transformers.load_model", return_value=loaded):
            model = MlflowOcrModel("models:/ocr-model/latest", gpu_device=0)
            model.load()
            assert model._model_architecture() == "glm"

    def test_architecture_defaults_to_qwen(self, mock_tensors):
        """Arquitetura desconhecida deve cair no fallback qwen."""
        loaded = _make_mlflow_loaded("UnknownArchitecture", mock_tensors)

        with patch("mlflow.transformers.load_model", return_value=loaded):
            model = MlflowOcrModel("models:/ocr-model/latest", gpu_device=0)
            model.load()
            assert model._model_architecture() == "qwen"

    def test_predict_qwen_returns_string(self, mock_tensors):
        """Predict com arquitetura qwen deve retornar string."""
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
        """Predict com arquitetura glm deve retornar string."""
        loaded = _make_mlflow_loaded("ChatGLMModel", mock_tensors)

        with patch("mlflow.transformers.load_model", return_value=loaded):
            model = MlflowOcrModel("models:/ocr-model/latest", gpu_device=0)
            model.load()

            img = Image.new("RGB", (100, 50), color="white")
            result = model.predict(img, max_tokens=128)

            assert isinstance(result, str)

    def test_device_with_gpu(self, mock_tensors):
        """device property deve usar gpu_device configurado."""
        loaded = _make_mlflow_loaded("Qwen2_5_VLForConditionalGeneration", mock_tensors)

        with patch("mlflow.transformers.load_model", return_value=loaded):
            model = MlflowOcrModel("models:/ocr-model/latest", gpu_device=1)
            model.load()
            # device do modelo mockado é "cpu", então retorna "cpu"
            # O fallback com gpu_device=1 só é usado se não houver GPU disponível
            # e _model estiver None
            model._model = None
            assert model.device == "cpu"  # sem CUDA disponível, fallback pra cpu

    def test_unload_clears_model(self, mock_tensors):
        """unload deve limpar modelo e processor."""
        loaded = _make_mlflow_loaded("Qwen2_5_VLForConditionalGeneration", mock_tensors)

        with patch("mlflow.transformers.load_model", return_value=loaded):
            model = MlflowOcrModel("models:/ocr-model/latest", gpu_device=0)
            model.load()
            assert model._model is not None
            assert model._processor is not None

            model.unload()
            assert model._model is None
            assert model._processor is None
