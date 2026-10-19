from collections.abc import Callable

from models.ocr.base import AbstractOcrModel
from models.ocr.dharma import DharmaOcrModel
from models.ocr.glm import GlmOcrModel
from models.ocr.mlflow_model import MlflowOcrModel


# key -> constructor. Every entry is called with the same kwargs below, so this
# table never inspects classes: an OCR model with a different constructor
# registers a small lambda instead.
_MODEL_FACTORIES: dict[str, Callable[..., AbstractOcrModel]] = {
    "qwen": DharmaOcrModel,
    "dharma": DharmaOcrModel,
    "glm": GlmOcrModel,
}


def create_model(
    model_name: str,
    cache_dir: str,
    gpu_device: int,
    model_uri: str = "",
    dtype: str = "auto",
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
) -> AbstractOcrModel:
    if model_uri:
        return MlflowOcrModel(model_uri, gpu_device=gpu_device)

    for key, factory in _MODEL_FACTORIES.items():
        if key in model_name.lower():
            return factory(
                model_name=model_name,
                cache_dir=cache_dir,
                gpu_device=gpu_device,
                dtype=dtype,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
            )

    supported = ", ".join(_MODEL_FACTORIES)
    raise ValueError(
        f"Unknown or unsupported model '{model_name}'. "
        f"Supported architectures: {supported}. "
        "Set ocr_model_uri to load from MLflow instead."
    )
