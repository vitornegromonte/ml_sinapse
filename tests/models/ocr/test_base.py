from PIL import Image

from models.ocr.base import AbstractOcrModel


class _StubModel(AbstractOcrModel):
    def __init__(self):
        self.calls: list[tuple] = []

    def load(self) -> None:
        pass

    def predict(self, image, max_tokens, prompt=None) -> str:
        self.calls.append((image, max_tokens, prompt))
        return f"text-{len(self.calls)}"

    def unload(self) -> None:
        pass

    @property
    def device(self) -> str:
        return "cpu"


class TestPredictBatchDefault:
    def test_default_calls_predict_once_per_image_in_order(self):
        """Arrange: a model that only implements predict(), and 3 images.
        Act: call predict_batch.
        Assert: one predict() per image, texts returned in input order, with
        max_tokens and prompt passed through.
        """
        model = _StubModel()
        images = [Image.new("RGB", (4, 4)) for _ in range(3)]

        result = model.predict_batch(images, 128, prompt="p")

        assert result == ["text-1", "text-2", "text-3"]
        assert [(c[0], c[1], c[2]) for c in model.calls] == [(i, 128, "p") for i in images]

    def test_default_with_no_images_returns_empty_list(self):
        """Arrange: a model that only implements predict().
        Act: call predict_batch with no images.
        Assert: an empty list, and predict() is never called.
        """
        model = _StubModel()

        result = model.predict_batch([], 128)

        assert result == []
        assert model.calls == []
