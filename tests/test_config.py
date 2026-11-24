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


class TestOcrLimits:
    def test_defaults(self):
        """Arrange: default settings.
        Act: read the batch size and predict timeout.
        Assert: 4 pages per model pass and 600 seconds.
        """
        settings = Settings(_env_file=None)

        assert settings.ocr_page_batch_size == 4
        assert settings.ocr_predict_timeout_seconds == 600

    @pytest.mark.parametrize("field, value", [
        ("ocr_page_batch_size", 0),
        ("ocr_page_batch_size", 33),
        ("ocr_predict_timeout_seconds", 0),
    ])
    def test_out_of_range_limits_are_rejected(self, field, value):
        """Arrange: a batch size or timeout outside its bound.
        Act: build the settings.
        Assert: validation fails, so no unbounded batch or timeout is possible.
        """
        with pytest.raises(ValidationError):
            Settings(_env_file=None, **{field: value})


class TestOcrGpuMemory:
    def test_default_is_15_percent(self):
        """Arrange: default settings with no override.
        Act: read ocr_gpu_memory_utilization.
        Assert: 0.15.
        """
        settings = Settings(_env_file=None)

        assert settings.ocr_gpu_memory_utilization == 0.15

    def test_default_context_cap_covers_a_page_and_its_output(self):
        """Arrange: default settings.
        Act: read ocr_max_model_len with the page and output limits.
        Assert: 24,576 is above the largest image (16,384 tokens) plus the
        default output cap, so no page is rejected, and below the native 128k.
        """
        settings = Settings(_env_file=None)

        assert settings.ocr_max_model_len == 24576
        assert 16384 + settings.ocr_max_tokens < settings.ocr_max_model_len < 128000

    def test_lower_value_is_accepted(self):
        """Arrange: a value below the cap.
        Act: build the settings.
        Assert: the value is kept.
        """
        settings = Settings(_env_file=None, ocr_gpu_memory_utilization=0.1)

        assert settings.ocr_gpu_memory_utilization == 0.1

    @pytest.mark.parametrize("value", [0.16, 0.5, 0.9, 1.0, 0, -0.1])
    def test_value_above_cap_or_not_positive_is_rejected(self, value):
        """Arrange: a utilization above 15% or not positive.
        Act: build the settings.
        Assert: validation fails, so the OCR engine can never take over the card.
        """
        with pytest.raises(ValidationError):
            Settings(_env_file=None, ocr_gpu_memory_utilization=value)

    def test_environment_cannot_raise_the_cap(self, monkeypatch):
        """Arrange: OCR_GPU_MEMORY_UTILIZATION=0.9 in the environment.
        Act: build the settings.
        Assert: validation fails, so the old 0.9 deploy value is caught at startup.
        """
        monkeypatch.setenv("OCR_GPU_MEMORY_UTILIZATION", "0.9")

        with pytest.raises(ValidationError):
            Settings(_env_file=None)


class TestOcrCheckpointSetting:
    def test_default_ttl_is_one_day(self):
        """Arrange: default settings.
        Act: read ocr_partial_ttl_seconds.
        Assert: 86,400 seconds.
        """
        assert Settings(_env_file=None).ocr_partial_ttl_seconds == 86400

    def test_zero_turns_the_checkpoint_off(self):
        """Arrange: a TTL of 0.
        Act: build the settings.
        Assert: 0 is accepted (it means "off").
        """
        assert Settings(_env_file=None, ocr_partial_ttl_seconds=0).ocr_partial_ttl_seconds == 0

    def test_negative_ttl_is_rejected(self):
        """Arrange: a negative TTL.
        Act: build the settings.
        Assert: validation fails.
        """
        with pytest.raises(ValidationError):
            Settings(_env_file=None, ocr_partial_ttl_seconds=-1)
