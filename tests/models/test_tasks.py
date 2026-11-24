"""The typed payloads and the reject policy they share (see models/tasks.py).

A payload that validates but has nothing to do is skipped by the worker with a
warning; one that does not validate raises, so it takes the retry → DLQ path
instead of disappearing.
"""
import pytest
from pydantic import ValidationError

from models.tasks import EmbeddingTask, LinkageTask, LlmTask, NerTask


class TestNerTask:
    def test_requires_text(self):
        """Arrange: a NER payload with no text.
        Act: validate it.
        Assert: ValidationError.
        """
        with pytest.raises(ValidationError):
            NerTask(document_id="d1")

    def test_document_id_defaults_to_empty(self):
        """Arrange: a valid NER payload without a document id.
        Act: read document_id.
        Assert: it defaults to an empty string.
        """
        assert NerTask(text="oi").document_id == ""


class TestLlmTask:
    def test_requires_document_id(self):
        """Arrange: an LLM payload with text but no document id.
        Act: validate it.
        Assert: ValidationError.
        """
        with pytest.raises(ValidationError):
            LlmTask(text="oi")

    def test_text_defaults_to_none_so_emptiness_is_the_workers_call(self):
        """Arrange: LLM payloads without text, or with text explicitly None.
        Act: build each task.
        Assert: text stays None — the worker decides whether that means skip.
        """
        assert LlmTask(document_id="d1").text is None
        assert LlmTask(document_id="d1", text=None).text is None


class TestEmbeddingTask:
    def test_requires_boletim_id(self):
        """Arrange: an embedding payload with no boletim_id.
        Act: validate it.
        Assert: ValidationError.
        """
        with pytest.raises(ValidationError):
            EmbeddingTask()

    def test_coerces_a_numeric_string(self):
        """Arrange: a boletim_id arriving as a numeric string from JSON.
        Act: build the embedding task.
        Assert: it is coerced to an integer.
        """
        assert EmbeddingTask(boletim_id="123").boletim_id == 123


class TestLinkageTask:
    def test_requires_both_ids(self):
        """Arrange: a linkage payload with only the origin BO.
        Act: validate it.
        Assert: ValidationError — both IDs are required.
        """
        with pytest.raises(ValidationError):
            LinkageTask(id_bo_origem=1)

    def test_pipeline_run_id_is_optional(self):
        """Arrange: a linkage payload with both BO IDs and no run id.
        Act: build the task.
        Assert: pipeline_run_id is None.
        """
        assert LinkageTask(id_bo_origem=1, id_bo_destino=2).pipeline_run_id is None


class TestPolicy:
    def test_validation_error_is_a_valueerror(self):
        """Arrange: pydantic's ValidationError.
        Act: check it against ValueError.
        Assert: it is a subclass, so base_worker's per-message handling treats
        it like the old ValueError/KeyError.
        """
        assert issubclass(ValidationError, ValueError)

    def test_unknown_fields_are_ignored_not_rejected(self):
        """Arrange: a payload carrying an extra field, as an OCR result
        forwarded to the LLM worker does.
        Act: build the task.
        Assert: the extra field is ignored and the known one is kept — an
        extra field must not become a dead-letter.
        """
        assert NerTask(text="oi", forwarded_from_ocr="x").text == "oi"
