from models.ocr.task import OcrTask


class TestOcrTask:
    def test_valid_url(self):
        """OcrTask com image_url deve reconhecer que tem imagem."""
        task = OcrTask(document_id="d1", image_url="http://example.com/img.png")
        assert task.has_image() is True
        assert task.document_id == "d1"
        assert task.image_url == "http://example.com/img.png"

    def test_valid_base64(self):
        """OcrTask com image_base64 deve reconhecer que tem imagem."""
        task = OcrTask(document_id="d2", image_base64="aW1hZ2VuYW1l==")
        assert task.has_image() is True

    def test_valid_path(self):
        """OcrTask com image_path deve reconhecer que tem imagem."""
        task = OcrTask(document_id="d3", image_path="/tmp/test.png")
        assert task.has_image() is True

    def test_empty_payload(self):
        """OcrTask sem nenhuma fonte de imagem deve retornar has_image=False."""
        task = OcrTask(document_id="d4")
        assert task.has_image() is False

    def test_document_id_default(self):
        """document_id deve ter string vazia como padrão."""
        task = OcrTask()
        assert task.document_id == ""
