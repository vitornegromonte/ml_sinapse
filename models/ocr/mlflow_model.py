from pathlib import Path

import torch
from PIL import Image

from models.ocr.base import AbstractOcrModel

SYSTEM_PROMPT = (
    "Você é um assistente de transcrição de documentos, "
    "sua função é extrair todo o texto de uma imagem"
)

USER_PROMPT = (
    "Extraia todo o texto e todas as tabelas deste documento. "
    "Transcreva todas as tabelas caso existam multiplas."
)


class MlflowOcrModel(AbstractOcrModel):
    def __init__(self, model_uri: str, gpu_device: int = 0):
        self._model_uri = model_uri
        self._gpu_device = gpu_device
        self._processor = None
        self._model = None

    def load(self) -> None:
        import mlflow.transformers

        loaded = mlflow.transformers.load_model(self._model_uri)

        self._model = loaded.model if hasattr(loaded, "model") else loaded
        self._processor = (
            loaded.processor
            if hasattr(loaded, "processor")
            else getattr(loaded, "image_processor", None)
            or getattr(loaded, "tokenizer", None)
            or loaded
        )

        if hasattr(self._model, "to"):
            self._model.to(self.device)
        if hasattr(self._model, "eval"):
            self._model.eval()

    def _model_architecture(self) -> str:
        class_name = type(self._model).__name__.lower()
        if "qwen" in class_name:
            return "qwen"
        if "glm" in class_name or "chatglm" in class_name:
            return "glm"
        return "qwen"

    def predict(self, image: Image.Image, max_tokens: int) -> str:
        arch = self._model_architecture()

        if arch == "glm":
            return self._predict_glm(image, max_tokens)
        return self._predict_qwen(image, max_tokens)

    def _predict_qwen(self, image: Image.Image, max_tokens: int) -> str:
        from qwen_vl_utils import process_vision_info

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": USER_PROMPT},
            ]},
        ]

        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        kwargs = dict(text=[text], images=image_inputs, padding=True, return_tensors="pt")
        if video_inputs:
            kwargs["videos"] = video_inputs

        inputs = self._processor(**kwargs).to(self.device)

        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, max_new_tokens=max_tokens)

        trimmed = [
            out[len(in_ids):] for in_ids, out in zip(inputs.input_ids, generated_ids)
        ]
        return self._processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    def _predict_glm(self, image: Image.Image, max_tokens: int) -> str:
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": (
                    "Transcreva todo o texto presente nesta imagem "
                    "com extrema precisão, no idioma português do Brasil."
                )},
            ]}
        ]
        text = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(text=text, images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

        input_len = inputs["input_ids"].shape[1]
        return self._processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

    def unload(self) -> None:
        self._model = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def device(self) -> str:
        if self._model is not None and hasattr(self._model, "device"):
            return str(self._model.device)
        if torch.cuda.is_available():
            return f"cuda:{self._gpu_device}"
        return "cpu"
