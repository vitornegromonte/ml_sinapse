import asyncio

import torch
from transformers import AutoTokenizer, AutoModel

from config import settings
from workers.base_worker import BaseWorker


class TokenizerWorker(BaseWorker):
    def __init__(self, cfg=settings):
        super().__init__(cfg.tokenizer_stream, cfg)
        self._tokenizer = None
        self._model = None
        self._device = "cpu"

    def _load_model(self) -> None:
        uri = self.cfg.tokenizer_model_uri
        if uri:
            self.logger.info("Loading tokenizer model from MLflow: %s", uri)
            import mlflow.transformers

            loaded = mlflow.transformers.load_model(uri)
            self._tokenizer = loaded.tokenizer if hasattr(loaded, "tokenizer") else loaded
            self._model = loaded.model if hasattr(loaded, "model") else loaded
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            if hasattr(self._model, "to"):
                self._model.to(self._device)
            self.logger.info("Tokenizer model loaded from MLflow")
            return

        path = self.cfg.tokenizer_model_path
        self.logger.info("Loading BERTimbau from %s", path)

        self._tokenizer = AutoTokenizer.from_pretrained(
            path,
            cache_dir=self.cfg.model_base_path,
        )
        self._model = AutoModel.from_pretrained(
            path,
            cache_dir=self.cfg.model_base_path,
            torch_dtype=(
                torch.float16 if torch.cuda.is_available() else torch.float32
            ),
        )
        self._device = (
            f"cuda:{self.cfg.tokenizer_gpu_device}"
            if torch.cuda.is_available()
            else "cpu"
        )
        self._model.to(self._device)
        self._model.eval()
        self.logger.info("BERTimbau loaded on %s", self._device)

    async def process(self, data: dict) -> None:
        if self._model is None:
            await asyncio.to_thread(self._load_model)

        text = data["text"]
        max_length = data.get("max_length", 512)
        return_embeddings = data.get("return_embeddings", False)

        def run(text: str) -> dict:
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            )
            tokens = self._tokenizer.convert_ids_to_tokens(
                inputs["input_ids"][0]
            )
            result = {
                "tokens": tokens,
                "input_ids": inputs["input_ids"][0].tolist(),
                "attention_mask": inputs["attention_mask"][0].tolist(),
                "n_tokens": len(tokens),
            }

            if return_embeddings:
                inputs = inputs.to(self._device)
                with torch.no_grad():
                    outputs = self._model(**inputs)
                result["embedding"] = (
                    outputs.last_hidden_state.mean(dim=1)
                    .squeeze()
                    .tolist()
                )
                result["embedding_dim"] = len(result["embedding"])

            return result

        result = await asyncio.to_thread(run, text)

        self.logger.info(
            "Tokenizer result: document_id=%s n_tokens=%d",
            data.get("document_id", "?"),
            result["n_tokens"],
        )
