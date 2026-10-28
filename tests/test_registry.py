import pytest

from models.ocr.registry import _MODEL_FACTORIES, create_model
from models.ocr.base import AbstractOcrModel
from models.ocr.dharma import DharmaOcrModel
from models.ocr.glm import GlmOcrModel
from models.ocr.mlflow_model import MlflowOcrModel


class TestRegistry:
    def test_create_dharma_by_name(self):
        """Arrange: the Dharma OCR model name.
        Act: create the model from it.
        Assert: a DharmaOcrModel is returned.
        """
        model = create_model("Dharma-AI/Dharma-OCR-LITE", "/cache", 0)
        assert isinstance(model, DharmaOcrModel)

    def test_create_dharma_by_qwen(self):
        """Arrange: a model name containing "qwen".
        Act: create the model from it.
        Assert: the Qwen family resolves to DharmaOcrModel.
        """
        model = create_model("Qwen/Qwen2-VL-7B", "/cache", 0)
        assert isinstance(model, DharmaOcrModel)

    def test_create_glm(self):
        """Arrange: a model name containing "glm".
        Act: create the model from it.
        Assert: a GlmOcrModel is returned.
        """
        model = create_model("THUDM/glm-4v-9b", "/cache", 0)
        assert isinstance(model, GlmOcrModel)

    def test_create_mlflow(self):
        """Arrange: a model URI and an otherwise irrelevant model name.
        Act: create the model.
        Assert: MlflowOcrModel wins when model_uri is set.
        """
        model = create_model("irrelevant", "/cache", 0, model_uri="models:/ocr-model/1")
        assert isinstance(model, MlflowOcrModel)

    def test_create_mlflow_ignores_name(self):
        """Arrange: a GLM model name and a model URI.
        Act: create the model.
        Assert: the name is ignored and MlflowOcrModel is returned.
        """
        model = create_model("glm-4v-9b", "/cache", 0, model_uri="models:/ocr-model/1")
        assert isinstance(model, MlflowOcrModel)

    def test_unknown_model_raises_error(self):
        """Arrange: a model name no factory matches.
        Act: create the model.
        Assert: ValueError naming the unsupported model.
        """
        with pytest.raises(ValueError, match="Unknown or unsupported"):
            create_model("some/unknown-model", "/cache", 0)

    def test_every_registered_factory_accepts_create_model_kwargs(self):
        """Arrange: every entry in the OCR model factory registry.
        Act: build each one through create_model with the same arguments.
        Assert: each returns an AbstractOcrModel — an incompatible constructor
        fails here, not at worker boot in production.
        """
        for key in _MODEL_FACTORIES:
            model = create_model(f"{key}-qualquer", "/cache", 0)
            assert isinstance(model, AbstractOcrModel)

    def test_mlflow_gpu_device_passed(self):
        """Arrange: a model URI and GPU device 2.
        Act: create the MLflow model.
        Assert: the instance is an MlflowOcrModel bound to device 2.
        """
        model = create_model("irrelevant", "/cache", 2, model_uri="models:/ocr-model/1")
        assert isinstance(model, MlflowOcrModel)
        assert model._gpu_device == 2
