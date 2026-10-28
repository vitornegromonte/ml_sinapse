from models.ocr.task import OcrTask


class TestOcrTask:
    def test_valid_url(self):
        """Arrange: an OcrTask carrying an image URL.
        Act: ask whether it has an image.
        Assert: yes, and the document id and URL are preserved.
        """
        task = OcrTask(document_id="d1", image_url="http://example.com/img.png")
        assert task.has_image() is True
        assert task.document_id == "d1"
        assert task.image_url == "http://example.com/img.png"

    def test_valid_base64(self):
        """Arrange: an OcrTask carrying an inline base64 image.
        Act: ask whether it has an image.
        Assert: yes.
        """
        task = OcrTask(document_id="d2", image_base64="aW1hZ2VuYW1l==")
        assert task.has_image() is True

    def test_valid_path(self):
        """Arrange: an OcrTask carrying a local image path.
        Act: ask whether it has an image.
        Assert: yes.
        """
        task = OcrTask(document_id="d3", image_path="/tmp/test.png")
        assert task.has_image() is True

    def test_empty_payload(self):
        """Arrange: an OcrTask with no image source.
        Act: ask whether it has an image.
        Assert: no.
        """
        task = OcrTask(document_id="d4")
        assert task.has_image() is False

    def test_document_id_default(self):
        """Arrange: an OcrTask built with no arguments.
        Act: read document_id.
        Assert: it defaults to an empty string.
        """
        task = OcrTask()
        assert task.document_id == ""
