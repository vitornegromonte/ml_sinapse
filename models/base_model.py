from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


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
