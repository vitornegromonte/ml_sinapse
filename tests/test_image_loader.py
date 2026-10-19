import base64
import io
import os
import tempfile
from unittest.mock import patch

import pytest
from PIL import Image

from services.image_loader import ImageLoader


class TestImageLoader:
    def test_from_url(self):
        """ImageLoader deve baixar imagem de URL e retornar PIL.Image."""
        loader = ImageLoader()
        # Cria um PNG em memória
        img = Image.new("RGB", (10, 10), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_data = buf.getvalue()

        with patch("requests.get") as mock_get:
            mock_get.return_value.raise_for_status.return_value = None
            mock_get.return_value.content = png_data

            result = loader.load({"image_url": "https://example.com/test.png"})

            assert isinstance(result, Image.Image)
            assert result.size == (10, 10)

    def test_from_base64(self):
        """ImageLoader deve decodificar base64 e retornar PIL.Image."""
        loader = ImageLoader()
        img = Image.new("RGB", (20, 30), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        result = loader.load({"image_base64": b64})

        assert isinstance(result, Image.Image)
        assert result.size == (20, 30)

    def test_from_path(self):
        """ImageLoader deve abrir arquivo dentro do allowed_root."""
        img = Image.new("RGB", (40, 50), color="green")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f, format="PNG")
            tmp_path = f.name

        try:
            loader = ImageLoader(allowed_root=os.path.dirname(tmp_path))
            result = loader.load({"image_path": tmp_path})
            assert isinstance(result, Image.Image)
            assert result.size == (40, 50)
        finally:
            os.unlink(tmp_path)

    def test_from_path_disabled_without_allowed_root(self):
        """Sem allowed_root, image_path é recusado — não é leitura arbitrária."""
        loader = ImageLoader()
        with pytest.raises(ValueError, match="disabled"):
            loader.load({"image_path": "/etc/passwd"})

    def test_from_path_rejects_traversal(self, tmp_path):
        root = tmp_path / "images"
        root.mkdir()
        loader = ImageLoader(allowed_root=root)

        with pytest.raises(ValueError, match="escapes"):
            loader.load({"image_path": "../../etc/passwd"})

    def test_from_path_rejects_absolute_path_outside_root(self, tmp_path):
        root = tmp_path / "images"
        root.mkdir()
        loader = ImageLoader(allowed_root=root)

        with pytest.raises(ValueError, match="escapes"):
            loader.load({"image_path": "/etc/passwd"})

    def test_from_path_rejects_symlink_escape(self, tmp_path):
        root = tmp_path / "images"
        root.mkdir()
        secret = tmp_path / "secret.txt"
        secret.write_text("nao deveria ser legivel")
        (root / "link.txt").symlink_to(secret)
        loader = ImageLoader(allowed_root=root)

        with pytest.raises(ValueError, match="escapes"):
            loader.load({"image_path": "link.txt"})

    def test_no_source_raises_value_error(self):
        """ImageLoader com dict vazio deve levantar ValueError."""
        loader = ImageLoader()
        with pytest.raises(ValueError, match="image_url"):
            loader.load({})
