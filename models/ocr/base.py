from abc import ABC, abstractmethod

from PIL import Image


class AbstractOcrModel(ABC):
    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def predict(self, image: Image.Image, max_tokens: int, prompt: str | None = None) -> str:
        ...

    def predict_batch(
        self, images: list[Image.Image], max_tokens: int, prompt: str | None = None
    ) -> list[str]:
        """OCR several images and return one text per image, in order.

        The default calls predict() one image at a time, so any model works
        unchanged. A model that can run images in one GPU call overrides it.
        """
        return [self.predict(image, max_tokens, prompt) for image in images]

    @abstractmethod
    def unload(self) -> None:
        ...

    @property
    @abstractmethod
    def device(self) -> str:
        ...
