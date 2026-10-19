import pytest

from models.ocr.registry import _MODEL_FACTORIES, create_model
from models.ocr.base import AbstractOcrModel
from models.ocr.dharma import DharmaOcrModel
from models.ocr.glm import GlmOcrModel
from models.ocr.mlflow_model import MlflowOcrModel


class TestRegistry:
    def test_create_dharma_by_name(self):
        """create_model com 'Dharma-AI/Dharma-OCR-LITE' deve retornar DharmaOcrModel."""
        model = create_model("Dharma-AI/Dharma-OCR-LITE", "/cache", 0)
        assert isinstance(model, DharmaOcrModel)

    def test_create_dharma_by_qwen(self):
        """create_model com nome contendo 'qwen' deve retornar DharmaOcrModel."""
        model = create_model("Qwen/Qwen2-VL-7B", "/cache", 0)
        assert isinstance(model, DharmaOcrModel)

    def test_create_glm(self):
        """create_model com nome contendo 'glm' deve retornar GlmOcrModel."""
        model = create_model("THUDM/glm-4v-9b", "/cache", 0)
        assert isinstance(model, GlmOcrModel)

    def test_create_mlflow(self):
        """create_model com model_uri definido deve retornar MlflowOcrModel."""
        model = create_model("irrelevant", "/cache", 0, model_uri="models:/ocr-model/1")
        assert isinstance(model, MlflowOcrModel)

    def test_create_mlflow_ignores_name(self):
        """Quando model_uri está definido, o nome do modelo é ignorado."""
        model = create_model("glm-4v-9b", "/cache", 0, model_uri="models:/ocr-model/1")
        assert isinstance(model, MlflowOcrModel)

    def test_unknown_model_raises_error(self):
        """Nome de modelo desconhecido deve levantar ValueError."""
        with pytest.raises(ValueError, match="Unknown or unsupported"):
            create_model("some/unknown-model", "/cache", 0)

    def test_every_registered_factory_accepts_create_model_kwargs(self):
        """create_model chama toda entrada com os mesmos kwargs (sem inspecionar
        classes) — uma entrada nova com construtor incompatível quebra aqui, não
        no boot de um worker em produção."""
        for key in _MODEL_FACTORIES:
            model = create_model(f"{key}-qualquer", "/cache", 0)
            assert isinstance(model, AbstractOcrModel)

    def test_mlflow_gpu_device_passed(self):
        """MlflowOcrModel deve receber gpu_device."""
        model = create_model("irrelevant", "/cache", 2, model_uri="models:/ocr-model/1")
        assert isinstance(model, MlflowOcrModel)
        assert model._gpu_device == 2
