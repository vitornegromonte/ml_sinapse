import asyncio
import io
import json
import logging

import pytest

from services.logging_setup import (
    TEXT_FORMAT,
    JsonFormatter,
    build_handler,
    configure_logging,
    context_from_payload,
    message_context,
)


@pytest.fixture
def capture():
    """A logger with a JSON handler that writes to a buffer. Returns (logger, lines())."""
    buffer = io.StringIO()
    handler = build_handler("json", {"stream": "ocr:tasks", "manifest_id": "abc123"})
    handler.stream = buffer
    logger = logging.getLogger(f"test.logging.{id(buffer)}")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    yield logger, lambda: [json.loads(line) for line in buffer.getvalue().splitlines()]
    logger.removeHandler(handler)


class TestContext:
    def test_fields_appear_inside_the_block_and_vanish_after(self, capture):
        """Arrange: a JSON logger.
        Act: log inside and outside a message_context block.
        Assert: the field is on the first line only.
        """
        logger, lines = capture

        with message_context(message_id="1-0"):
            logger.info("inside")
        logger.info("outside")

        inside, outside = lines()
        assert inside["message_id"] == "1-0"
        assert "message_id" not in outside

    def test_nested_contexts_merge_and_unwind(self, capture):
        """Arrange: two nested blocks.
        Act: log in the inner block and then in the outer one.
        Assert: the inner line has both fields and the outer line only its own.
        """
        logger, lines = capture

        with message_context(message_id="1-0"):
            with message_context(document_id="bo-9"):
                logger.info("inner")
            logger.info("outer")

        inner, outer = lines()
        assert inner["message_id"] == "1-0" and inner["document_id"] == "bo-9"
        assert outer["message_id"] == "1-0" and "document_id" not in outer

    def test_none_values_are_dropped(self, capture):
        """Arrange: a block with one value and one None.
        Act: log.
        Assert: only the real value is in the line.
        """
        logger, lines = capture

        with message_context(document_id="bo-1", pipeline_run_id=None):
            logger.info("x")

        assert "pipeline_run_id" not in lines()[0] and lines()[0]["document_id"] == "bo-1"

    def test_the_context_unwinds_even_when_the_block_raises(self, capture):
        """Arrange: a block that raises.
        Act: run it, then log.
        Assert: the field does not leak into later lines.
        """
        logger, lines = capture

        with pytest.raises(ValueError):
            with message_context(message_id="9-9"):
                raise ValueError("boom")
        logger.info("after")

        assert "message_id" not in lines()[0]

    async def test_the_context_follows_work_into_a_thread(self, capture):
        """Arrange: a message_context block.
        Act: log from asyncio.to_thread inside it (the OCR worker logs from threads).
        Assert: the line has the field.
        """
        logger, lines = capture

        with message_context(document_id="bo-thread"):
            await asyncio.to_thread(logger.info, "from a thread")

        assert lines()[0]["document_id"] == "bo-thread"

    async def test_the_context_follows_work_into_a_task(self, capture):
        """Arrange: a message_context block that creates a task.
        Act: log from the task.
        Assert: the line has the field (the lease heartbeat is such a task).
        """
        logger, lines = capture

        async def work():
            logger.info("from a task")

        with message_context(document_id="bo-task"):
            await asyncio.create_task(work())

        assert lines()[0]["document_id"] == "bo-task"


class TestContextFromPayload:
    def test_only_the_id_keys_are_taken(self):
        """Arrange: a payload with ids, text, and an image.
        Act: build the context.
        Assert: the ids are in it, and the text and image are not.
        """
        context = context_from_payload(
            "1-0", {"document_id": "bo-1", "pipeline_run_id": "run-7", "text": "SECRET", "image_base64": "AAAA"}
        )

        assert context == {"message_id": "1-0", "document_id": "bo-1", "pipeline_run_id": "run-7"}

    def test_numbers_become_strings_and_empty_values_are_skipped(self):
        """Arrange: a payload with an integer id, an empty string and a None.
        Act: build the context.
        Assert: the integer is a string, and the empty and None are left out.
        """
        context = context_from_payload("2-0", {"boletim_id": 42, "document_id": "", "pipeline_run_id": None})

        assert context == {"message_id": "2-0", "boletim_id": "42"}

    def test_a_linkage_pair_is_covered(self):
        """Arrange: a linkage task payload.
        Act: build the context.
        Assert: both BO ids are in it.
        """
        context = context_from_payload("3-0", {"id_bo_origem": 1, "id_bo_destino": 2})

        assert context["id_bo_origem"] == "1" and context["id_bo_destino"] == "2"


class TestJsonFormat:
    def test_a_line_is_one_json_object_with_the_fixed_fields(self, capture):
        """Arrange: a JSON logger with fixed stream and manifest fields.
        Act: log a message with arguments.
        Assert: one parseable object with the level, logger, message, and fixed fields.
        """
        logger, lines = capture

        logger.warning("took %d seconds", 3)

        line = lines()[0]
        assert line["level"] == "WARNING" and line["message"] == "took 3 seconds"
        assert line["stream"] == "ocr:tasks" and line["manifest_id"] == "abc123"
        assert line["ts"].endswith("+00:00")

    def test_non_ascii_text_is_kept(self, capture):
        """Arrange: a JSON logger.
        Act: log Portuguese text.
        Assert: the accents are kept, not escaped.
        """
        logger, _ = capture
        stream = logger.handlers[0].stream

        logger.info("ocorrência")

        assert "ocorrência" in stream.getvalue()

    def test_an_exception_is_included(self, capture):
        """Arrange: a JSON logger.
        Act: log inside an except block with exc_info.
        Assert: the line has an exception field with the traceback.
        """
        logger, lines = capture

        try:
            raise ValueError("bad payload")
        except ValueError:
            logger.error("failed", exc_info=True)

        assert "ValueError: bad payload" in lines()[0]["exception"]

    def test_formatter_works_on_a_bare_record(self):
        """Arrange: a record that never passed through the context filter.
        Act: format it.
        Assert: valid JSON with the message.
        """
        record = logging.LogRecord("x", logging.INFO, __file__, 1, "plain", None, None)

        assert json.loads(JsonFormatter().format(record))["message"] == "plain"


class TestTextFormat:
    def test_the_default_format_is_the_one_the_workers_always_used(self):
        """Arrange: the old basicConfig format string.
        Act: compare it with TEXT_FORMAT.
        Assert: identical, so log parsers keep working.
        """
        assert TEXT_FORMAT == "%(asctime)s [%(name)s] %(levelname)s: %(message)s"

    def test_text_lines_carry_no_json_and_no_extra_fields(self):
        """Arrange: a text handler with a context active.
        Act: log a line.
        Assert: it matches the old layout and does not show the context.
        """
        buffer = io.StringIO()
        handler = build_handler("text", {"stream": "ocr:tasks"})
        handler.stream = buffer
        logger = logging.getLogger("test.text.format")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        with message_context(document_id="bo-1"):
            logger.info("hello")
        logger.removeHandler(handler)

        line = buffer.getvalue().strip()
        assert line.endswith("[test.text.format] INFO: hello")
        assert "bo-1" not in line and "{" not in line


class TestConfigureLogging:
    def test_it_installs_one_handler_once_and_sets_the_level(self):
        """Arrange: a root logger with no handlers (saved and restored).
        Act: configure twice.
        Assert: one handler, and the level of the last call.
        """
        root = logging.getLogger()
        saved, level = root.handlers[:], root.level
        root.handlers = []
        try:
            configure_logging("WARNING", "json")
            configure_logging("DEBUG", "json")

            assert len(root.handlers) == 1
            assert root.level == logging.DEBUG
        finally:
            root.handlers, root.level = saved, level

    def test_it_leaves_an_existing_handler_alone(self):
        """Arrange: a root logger that already has a handler (as under pytest or a host app).
        Act: configure.
        Assert: no second handler is added, like logging.basicConfig.
        """
        root = logging.getLogger()
        saved, level = root.handlers[:], root.level
        existing = logging.NullHandler()
        root.handlers = [existing]
        try:
            configure_logging("INFO", "json")

            assert root.handlers == [existing]
        finally:
            root.handlers, root.level = saved, level
