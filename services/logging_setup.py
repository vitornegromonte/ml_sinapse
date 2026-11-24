"""Logging for the workers: the same text format as before, or JSON with context.

The text format is the one the workers always used. With LOG_FORMAT=json every
line is one JSON object, and it carries the ids of the message being handled
(message id, pipeline run id, document id) and the model recipe of the worker.
That lets a log search follow one document through the OCR, LLM and linkage
workers (ADR-015).

The context lives in a ContextVar, so it follows the code into asyncio tasks and
into threads started with asyncio.to_thread. Only ids go in: never document text.
"""
import contextlib
import contextvars
import datetime
import json
import logging
from collections.abc import Iterator

TEXT_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"

# The payload keys that identify the work. A key that is absent is left out.
CONTEXT_KEYS = ("pipeline_run_id", "document_id", "boletim_id", "id_bo_origem", "id_bo_destino")

_context: contextvars.ContextVar[dict] = contextvars.ContextVar("log_context", default={})


@contextlib.contextmanager
def message_context(**fields) -> Iterator[None]:
    """Add `fields` to every log line written inside the block. None values are dropped."""
    merged = {**_context.get(), **{k: v for k, v in fields.items() if v is not None}}
    token = _context.set(merged)
    try:
        yield
    finally:
        _context.reset(token)


def context_from_payload(message_id: str, data: dict) -> dict:
    """The log context of one stream message: its id and the ids in its payload."""
    context = {"message_id": message_id}
    for key in CONTEXT_KEYS:
        value = data.get(key)
        if value not in (None, ""):
            context[key] = str(value)
    return context


class ContextFilter(logging.Filter):
    """Copies the worker's fixed fields and the current context onto each record."""

    def __init__(self, fixed: dict | None = None):
        super().__init__()
        self._fixed = dict(fixed or {})

    def filter(self, record: logging.LogRecord) -> bool:
        record.ml_context = {**self._fixed, **_context.get()}
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        document = {
            "ts": datetime.datetime.fromtimestamp(record.created, datetime.timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **getattr(record, "ml_context", {}),
        }
        if record.exc_info:
            document["exception"] = self.formatException(record.exc_info)
        return json.dumps(document, ensure_ascii=False, default=str)


def build_handler(fmt: str = "text", fixed: dict | None = None) -> logging.Handler:
    """A stream handler with the context filter and the chosen format."""
    handler = logging.StreamHandler()
    handler.addFilter(ContextFilter(fixed))
    handler.setFormatter(JsonFormatter() if fmt == "json" else logging.Formatter(TEXT_FORMAT))
    return handler


def configure_logging(level: str, fmt: str = "text", fixed: dict | None = None) -> None:
    """Set up the root logger, as logging.basicConfig did: only if it has no handler yet."""
    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(build_handler(fmt, fixed))
    root.setLevel(getattr(logging, level))
