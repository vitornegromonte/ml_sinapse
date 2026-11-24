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


class TestLlmSettings:
    def test_default_model_is_s5_nano(self):
        """Arrange: default settings.
        Act: read llm_model_path.
        Assert: maritaca-ai/S5-nano.
        """
        assert Settings(_env_file=None).llm_model_path == "maritaca-ai/S5-nano"

    def test_default_turns_thinking_off(self):
        """Arrange: default settings.
        Act: read llm_chat_template_kwargs.
        Assert: enable_thinking is False, so summaries hold no reasoning text.
        """
        settings = Settings(_env_file=None)

        assert settings.llm_chat_template_kwargs == {"enable_thinking": False}

    def test_default_sampling_is_the_instruct_recipe_with_a_token_cap(self):
        """Arrange: default settings.
        Act: read llm_sampling_params.
        Assert: the S5-nano Instruct values are set, and max_tokens is well
        above vLLM's own default of 16, which cuts summaries off.
        """
        params = Settings(_env_file=None).llm_sampling_params

        assert params["temperature"] == 0.7
        assert params["top_p"] == 0.8
        assert params["top_k"] == 20
        assert params["presence_penalty"] == 1.5
        assert params["max_tokens"] >= 1024

    def test_default_engine_is_text_only_and_bounded(self):
        """Arrange: default settings.
        Act: read llm_engine_kwargs.
        Assert: the vision encoder is skipped, and max_num_seqs is capped. vLLM
        0.22.1 does not start S5-nano with its default of 1024 in this memory.
        """
        kwargs = Settings(_env_file=None).llm_engine_kwargs

        assert kwargs["language_model_only"] is True
        assert kwargs["max_num_seqs"] <= 128

    def test_default_gpu_share_is_the_measured_quarter(self):
        """Arrange: default settings.
        Act: read llm_gpu_memory_utilization and llm_max_model_len.
        Assert: 0.25 of the card with the model's native context.
        """
        settings = Settings(_env_file=None)

        assert settings.llm_gpu_memory_utilization == 0.25
        assert settings.llm_max_model_len is None

    def test_sampling_params_can_be_set_from_the_environment(self, monkeypatch):
        """Arrange: LLM_SAMPLING_PARAMS set to a JSON object.
        Act: build the settings.
        Assert: the object replaces the default.
        """
        monkeypatch.setenv("LLM_SAMPLING_PARAMS", '{"max_tokens": 64, "temperature": 0.0}')

        params = Settings(_env_file=None).llm_sampling_params

        assert params == {"max_tokens": 64, "temperature": 0.0}

    def test_model_path_can_be_set_from_the_environment(self, monkeypatch):
        """Arrange: LLM_MODEL_PATH set to another model.
        Act: build the settings.
        Assert: the environment wins over the default.
        """
        monkeypatch.setenv("LLM_MODEL_PATH", "org/other-model")

        assert Settings(_env_file=None).llm_model_path == "org/other-model"


class TestObservabilitySettings:
    def test_defaults_keep_everything_off_and_the_log_format_unchanged(self):
        """Arrange: default settings.
        Act: read the observability settings.
        Assert: no metrics port, the text log format, and the documented periods.
        """
        settings = Settings(_env_file=None)

        assert settings.metrics_port == 0
        assert settings.log_format == "text"
        assert settings.metrics_refresh_seconds == 15
        assert settings.health_grace_seconds == 30

    def test_json_is_a_valid_log_format(self):
        """Arrange: log_format json.
        Act: build the settings.
        Assert: accepted.
        """
        assert Settings(_env_file=None, log_format="json").log_format == "json"

    @pytest.mark.parametrize("field, value", [
        ("log_format", "xml"),
        ("metrics_port", -1),
        ("metrics_port", 70000),
        ("metrics_refresh_seconds", 0),
        ("health_grace_seconds", -1),
    ])
    def test_invalid_values_are_rejected(self, field, value):
        """Arrange: a value outside the allowed range or set.
        Act: build the settings.
        Assert: validation fails.
        """
        with pytest.raises(ValidationError):
            Settings(_env_file=None, **{field: value})

    def test_the_port_can_be_set_from_the_environment(self, monkeypatch):
        """Arrange: METRICS_PORT and LOG_FORMAT in the environment (as compose sets them).
        Act: build the settings.
        Assert: both are read.
        """
        monkeypatch.setenv("METRICS_PORT", "9100")
        monkeypatch.setenv("LOG_FORMAT", "json")

        settings = Settings(_env_file=None)

        assert settings.metrics_port == 9100 and settings.log_format == "json"
