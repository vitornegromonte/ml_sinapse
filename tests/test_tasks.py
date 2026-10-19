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
        with pytest.raises(ValidationError):
            NerTask(document_id="d1")

    def test_document_id_defaults_to_empty(self):
        assert NerTask(text="oi").document_id == ""


class TestLlmTask:
    def test_requires_document_id(self):
        with pytest.raises(ValidationError):
            LlmTask(text="oi")

    def test_text_defaults_to_none_so_emptiness_is_the_workers_call(self):
        assert LlmTask(document_id="d1").text is None
        assert LlmTask(document_id="d1", text=None).text is None


class TestEmbeddingTask:
    def test_requires_boletim_id(self):
        with pytest.raises(ValidationError):
            EmbeddingTask()

    def test_coerces_a_numeric_string(self):
        assert EmbeddingTask(boletim_id="123").boletim_id == 123


class TestLinkageTask:
    def test_requires_both_ids(self):
        with pytest.raises(ValidationError):
            LinkageTask(id_bo_origem=1)

    def test_pipeline_run_id_is_optional(self):
        assert LinkageTask(id_bo_origem=1, id_bo_destino=2).pipeline_run_id is None


class TestPolicy:
    def test_validation_error_is_a_valueerror(self):
        """base_worker's per-message error handling must see it the same way
        it saw the old ValueError/KeyError."""
        assert issubclass(ValidationError, ValueError)

    def test_unknown_fields_are_ignored_not_rejected(self):
        """Payloads legitimately carry extra fields (e.g. an OCR result
        forwarded to the LLM worker) — that must not become a dead-letter."""
        assert NerTask(text="oi", forwarded_from_ocr="x").text == "oi"
