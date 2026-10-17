from models.ocr.base import AbstractOcrModel
from models.ocr.dharma import DharmaOcrModel
from models.ocr.glm import GlmOcrModel
from models.ocr.mlflow_model import MlflowOcrModel
from models.ocr.vllm_chat_base import VllmChatOcrModel


_MODEL_CLASSES: dict[str, type[AbstractOcrModel]] = {
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

    for key, cls in _MODEL_CLASSES.items():
        if key in model_name.lower():
            if issubclass(cls, VllmChatOcrModel):
                # Every vLLM-chat-backed OCR model shares this constructor
                # shape — checking the base class (not a specific model)
                # means a future third one needs no change here.
                return cls(
                    model_name=model_name,
                    cache_dir=cache_dir,
                    gpu_device=gpu_device,
                    dtype=dtype,
                    gpu_memory_utilization=gpu_memory_utilization,
                    max_model_len=max_model_len,
                )
            return cls(model_name=model_name, cache_dir=cache_dir, gpu_device=gpu_device)

    supported = ", ".join(_MODEL_CLASSES)
    raise ValueError(
        f"Unknown or unsupported model '{model_name}'. "
        f"Supported architectures: {supported}. "
        "Set ocr_model_uri to load from MLflow instead."
    )
