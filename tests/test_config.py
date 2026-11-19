import pytest
from pydantic import ValidationError

from config import Settings


class TestOcrSettings:
    def test_default_ocr_model_is_dharma(self):
        """Arrange: default settings with no OCR_MODEL_NAME override.
        Act: read ocr_model_name.
        Assert: it is Dharma-OCR-LITE.
        """
        settings = Settings(_env_file=None)

        assert settings.ocr_model_name == "Dharma-AI/Dharma-OCR-LITE"

    def test_default_render_dpi_is_200(self):
        """Arrange: default settings with no OCR_RENDER_DPI override.
        Act: read ocr_render_dpi.
        Assert: it is 200.
        """
        settings = Settings(_env_file=None)

        assert settings.ocr_render_dpi == 200

    @pytest.mark.parametrize("dpi", [71, 401, 0, -150])
    def test_render_dpi_out_of_range_is_rejected(self, dpi):
        """Arrange: a DPI outside the 72-400 bound.
        Act: build the settings.
        Assert: validation fails, so a bad value cannot reach the renderer.
        """
        with pytest.raises(ValidationError):
            Settings(_env_file=None, ocr_render_dpi=dpi)

    def test_render_dpi_reads_environment(self, monkeypatch):
        """Arrange: OCR_RENDER_DPI set in the environment.
        Act: build the settings.
        Assert: the value comes from the environment.
        """
        monkeypatch.setenv("OCR_RENDER_DPI", "300")

        settings = Settings(_env_file=None)

        assert settings.ocr_render_dpi == 300
