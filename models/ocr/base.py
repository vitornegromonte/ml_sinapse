from abc import ABC, abstractmethod

from PIL import Image


class AbstractOcrModel(ABC):
    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def predict(self, image: Image.Image, max_tokens: int, prompt: str | None = None) -> str:
        ...

    @abstractmethod
    def unload(self) -> None:
        ...

    @property
    @abstractmethod
    def device(self) -> str:
        ...
