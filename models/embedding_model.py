"""Text encoder for BO embeddings.

Only SentenceTransformerEncoder exists — the Protocol seam that used to be
declared here had no second implementation and nothing depended on it. The
linkage classifier (Maynara_cluster_v0/modelo_v1_classificador.joblib) only
cares about the resulting vector, so another encoder (e.g. one served via
vLLM's --task embed) can be added if and when one is actually validated here.

CRITICAL: the classifier was calibrated against
sentence-transformers.SentenceTransformer("neuralmind/bert-base-portuguese-
cased").encode(...) specifically — that library's own pooling (mean pooling
over token embeddings, L2-normalized) is part of what "the embedding space"
means here. A different pooling strategy over the same base model produces a
numerically different space and silently invalidates score_par's sim_texto
component — no error, just quietly wrong scores. Do not swap encoders
without re-validating (or retraining) the classifier against the new space.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class SentenceTransformerEncoder:
    """Wraps sentence-transformers — see module docstring for why this
    specific library/pooling combination is load-bearing, not incidental."""

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None

    def load(self) -> None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading SentenceTransformer(%s) on %s", self.model_name, self.device)
        self._model = SentenceTransformer(self.model_name, device=self.device)
        logger.info(
            "Encoder loaded, embedding dim=%d",
            self._model.get_embedding_dimension(),
        )

    def unload(self) -> None:
        self._model = None
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Encoder not loaded. Call load() first.")
        return self._model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=min(32, len(texts)) if texts else 1,
            show_progress_bar=False,
        )
