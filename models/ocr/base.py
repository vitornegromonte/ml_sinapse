"""The interface every OCR model implements.

`OcrWorker` talks only to this interface. The concrete models live next to it:
`VllmChatOcrModel` (Dharma, GLM) and `MlflowOcrModel`. `create_model` in
`registry.py` picks one.
"""
from abc import ABC, abstractmethod

from PIL import Image


class AbstractOcrModel(ABC):
    """An OCR model that turns a page image into text.

    Lifecycle: `load()` once, then any number of `predict()` / `predict_batch()`
    calls, then `unload()`. The calls block, so callers run them in a thread.
    """

    @abstractmethod
    def load(self) -> None:
        """Load the weights and allocate GPU memory."""

    @abstractmethod
    def predict(self, image: Image.Image, max_tokens: int, prompt: str | None = None) -> str:
        """Return the text of one RGB image.

        `prompt` replaces the model's default user prompt when given.
        """

    def predict_batch(
        self, images: list[Image.Image], max_tokens: int, prompt: str | None = None
    ) -> list[str]:
        """Return one text per image, in the same order as `images`.

        The default calls `predict()` once per image, so any model works
        unchanged. A model that can run several images in one GPU call
        overrides this.
        """
        return [self.predict(image, max_tokens, prompt) for image in images]

    @abstractmethod
    def unload(self) -> None:
        """Release the model and its GPU memory."""

    @property
    @abstractmethod
    def device(self) -> str:
        """The device the model runs on, for example "cuda:0" or "cpu"."""
