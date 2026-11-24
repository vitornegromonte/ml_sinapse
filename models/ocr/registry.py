"""Choose the OCR model class from the configured model name."""
from collections.abc import Callable

from models.ocr.base import AbstractOcrModel
from models.ocr.dharma import DharmaOcrModel
from models.ocr.glm import GlmOcrModel
from models.ocr.mlflow_model import MlflowOcrModel

# Lowercase part of a model name -> class. The first key found in the name
# wins, in the order below. Every class is called with the same keyword
# arguments (see create_model), so a model with a different constructor
# registers a small lambda here.
_MODEL_FACTORIES: dict[str, Callable[..., AbstractOcrModel]] = {
    "qwen": DharmaOcrModel,
    "dharma": DharmaOcrModel,
    "glm": GlmOcrModel,
}


def model_class_for(model_name: str) -> Callable[..., AbstractOcrModel] | None:
    """The class create_model would build for `model_name`, or None if unknown.

    It builds nothing and loads nothing, so it is safe for code that only needs
    the class settings (for example the prompts, in the model manifest).
    """
    name = model_name.lower()
    for key, factory in _MODEL_FACTORIES.items():
        if key in name:
            return factory
    return None


def create_model(
    model_name: str,
    cache_dir: str,
    gpu_device: int,
    model_uri: str = "",
    dtype: str = "auto",
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
    revision: str | None = None,
) -> AbstractOcrModel:
    """Build the OCR model, without loading it.

    A `model_uri` wins over `model_name`: the model then comes from the MLflow
    registry and the vLLM settings, including `revision`, do not apply.
    Otherwise the class is chosen from `model_name`, and `revision` pins the
    Hugging Face commit. An unknown name raises ValueError.
    """
    if model_uri:
        return MlflowOcrModel(model_uri, gpu_device=gpu_device)

    name = model_name.lower()
    for key, factory in _MODEL_FACTORIES.items():
        if key in name:
            return factory(
                model_name=model_name,
                cache_dir=cache_dir,
                gpu_device=gpu_device,
                dtype=dtype,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                revision=revision,
            )

    supported = ", ".join(_MODEL_FACTORIES)
    raise ValueError(
        f"Unknown or unsupported model '{model_name}'. "
        f"Supported architectures: {supported}. "
        "Set ocr_model_uri to load from MLflow instead."
    )
