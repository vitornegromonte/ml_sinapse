from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import mlflow

logger = logging.getLogger(__name__)


class BaseModelWrapper(ABC):
    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def predict(self, inputs: Any, **kwargs: Any) -> Any:
        ...

    @abstractmethod
    def unload(self) -> None:
        ...

    @staticmethod
    def resolve_model_uri(model_uri: str, default_path: str, flavor: str = "pyfunc") -> str:
        """Return model_uri if set, else fall back to local path."""
        if model_uri:
            return model_uri
        logger.info("No MLflow URI set, using local path: %s", default_path)
        return default_path

    @staticmethod
    def load_from_registry(model_uri: str) -> Any:
        """Load a model from the MLflow Model Registry.

        Supports ``models:/name/stage`` and ``runs:/run_id/path`` formats.
        Falls back to ``mlflow.pyfunc`` unless a specific flavor is needed.
        """
        mlflow.set_tracking_uri(mlflow.tracking.get_tracking_uri())
        logger.info("Loading model from MLflow registry: %s", model_uri)
        return mlflow.pyfunc.load_model(model_uri)

    @staticmethod
    def register_model(
        local_model_path: str,
        registered_name: str,
        run_id: str | None = None,
        artifact_path: str = "model",
    ) -> str:
        """Register a model in the MLflow Model Registry and return its URI."""
        result = mlflow.register_model(
            model_uri=f"runs:/{run_id}/{artifact_path}" if run_id else local_model_path,
            name=registered_name,
        )
        logger.info(
            "Registered model '%s' version %s",
            registered_name,
            result.version,
        )
        return f"models:/{registered_name}/{result.version}"
