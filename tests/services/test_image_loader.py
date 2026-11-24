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
        """Arrange: an in-memory PNG and requests.get patched to return it.
        Act: load from image_url.
        Assert: a PIL image of the original size is returned.
        """
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
        """Arrange: an in-memory PNG encoded as base64.
        Act: load from image_base64.
        Assert: a PIL image of the original size is returned.
        """
        loader = ImageLoader()
        img = Image.new("RGB", (20, 30), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        result = loader.load({"image_base64": b64})

        assert isinstance(result, Image.Image)
        assert result.size == (20, 30)

    def test_from_path(self):
        """Arrange: a temporary PNG inside the allowed root.
        Act: load from image_path.
        Assert: a PIL image of the original size is returned.
        """
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
        """Arrange: a loader with no allowed_root.
        Act: load an absolute image_path.
        Assert: ValueError — local paths are disabled, not arbitrary reads.
        """
        loader = ImageLoader()
        with pytest.raises(ValueError, match="disabled"):
            loader.load({"image_path": "/etc/passwd"})

    def test_from_path_rejects_traversal(self, tmp_path):
        """Arrange: a loader whose allowed root is tmp_path/images.
        Act: load ../../etc/passwd.
        Assert: ValueError — traversal escapes the root.
        """
        root = tmp_path / "images"
        root.mkdir()
        loader = ImageLoader(allowed_root=root)

        with pytest.raises(ValueError, match="escapes"):
            loader.load({"image_path": "../../etc/passwd"})

    def test_from_path_rejects_absolute_path_outside_root(self, tmp_path):
        """Arrange: a loader whose allowed root is tmp_path/images.
        Act: load /etc/passwd.
        Assert: ValueError — an absolute path outside the root is refused.
        """
        root = tmp_path / "images"
        root.mkdir()
        loader = ImageLoader(allowed_root=root)

        with pytest.raises(ValueError, match="escapes"):
            loader.load({"image_path": "/etc/passwd"})

    def test_from_path_rejects_symlink_escape(self, tmp_path):
        """Arrange: a symlink inside the allowed root pointing outside it.
        Act: load through the symlink.
        Assert: ValueError — resolve() follows the link and the containment
        check catches it.
        """
        root = tmp_path / "images"
        root.mkdir()
        secret = tmp_path / "secret.txt"
        secret.write_text("nao deveria ser legivel")
        (root / "link.txt").symlink_to(secret)
        loader = ImageLoader(allowed_root=root)

        with pytest.raises(ValueError, match="escapes"):
            loader.load({"image_path": "link.txt"})

    def test_no_source_raises_value_error(self):
        """Arrange: a loader and an empty payload.
        Act: load it.
        Assert: ValueError naming the missing image_url.
        """
        loader = ImageLoader()
        with pytest.raises(ValueError, match="image_url"):
            loader.load({})
