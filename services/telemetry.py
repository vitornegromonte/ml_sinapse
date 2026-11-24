"""Worker metrics and health: what an operator needs to see from outside.

One WorkerTelemetry per worker process holds the Prometheus metrics and the
liveness state. A small HTTP server exposes three paths:

    /metrics   Prometheus text format
    /healthz   200 while the worker loop makes progress, 503 when it stalls
    /readyz    200 while the loop runs and Redis answers

It is off unless METRICS_PORT is set (ADR-015). Rules that keep it safe:

  * Observability must never break processing. Every recording method catches
    its own errors and logs them at debug level.
  * Metrics hold no document text and no ids of documents. The only labels are
    the stream, an outcome, and (on one info metric) the model recipe.
  * Liveness counts the main loop and the lease heartbeat, so a long model
    call keeps the worker healthy, and a blocked event loop does not.
"""
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

logger = logging.getLogger(__name__)

# Seconds. A summary or a scanned PDF can take minutes; a cluster run, longer.
_DURATION_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600, 1200)
_BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64)

OUTCOMES = ("success", "retry", "dead_letter", "permanent")


def _safe(method):
    """Run a recording method without ever letting its error out."""

    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as exc:  # observability must not break processing
            logger.debug("Telemetry call %s failed: %s", method.__name__, exc)

    wrapper.__name__ = method.__name__
    wrapper.__doc__ = method.__doc__
    return wrapper


class WorkerTelemetry:
    """Metrics and health state of one worker process.

    Args:
        stream: the stream the worker consumes. It is the only label on most metrics.
        manifest: the worker's StageManifest, if it has one. It becomes the
            ml_model_info metric, so a dashboard can show which recipe runs.
        stale_after_seconds: how long the loop may go without a tick before the
            worker counts as stalled. It must exceed the idle poll timeout.
        clock: a monotonic clock. A test passes a fake one.
    """

    def __init__(self, stream: str, manifest=None, stale_after_seconds: float = 80.0, clock=time.monotonic):
        self.stream = stream
        self._stale_after = stale_after_seconds
        self._clock = clock
        now = clock()
        self._last_tick = now
        self._last_redis_ok = now
        self._model_loaded = False
        self.registry = CollectorRegistry()

        labels = ["stream"]
        self._messages = Counter(
            "ml_messages_total", "Messages settled, by outcome.", ["stream", "outcome"], registry=self.registry
        )
        self._reclaimed = Counter(
            "ml_messages_reclaimed_total", "Pending messages taken over by XAUTOCLAIM.", labels, registry=self.registry
        )
        self._duration = Histogram(
            "ml_process_duration_seconds", "Time of one process_batch call (one observation per call).",
            labels, buckets=_DURATION_BUCKETS, registry=self.registry,
        )
        self._batch_size = Histogram(
            "ml_batch_size", "Messages in one process_batch call.", labels, buckets=_BATCH_BUCKETS, registry=self.registry
        )
        self._queue_length = Gauge("ml_queue_length", "Entries in the task stream.", labels, registry=self.registry)
        self._queue_pending = Gauge(
            "ml_queue_pending", "Delivered but not yet acked messages of the group.", labels, registry=self.registry
        )
        self._dead_letter_length = Gauge(
            "ml_dead_letter_length", "Entries in the dead-letter stream.", labels, registry=self.registry
        )
        self._model_loaded_gauge = Gauge(
            "ml_model_loaded", "1 when the worker's model is loaded.", labels, registry=self.registry
        )
        self._model_info = Gauge(
            "ml_model_info", "The model recipe the worker runs (value is always 1).",
            ["stream", "manifest_id", "model"], registry=self.registry,
        )
        self._last_message = Gauge(
            "ml_last_message_timestamp_seconds", "Unix time of the last message settled.", labels, registry=self.registry
        )
        self._result_stream_length = Gauge(
            "ml_result_stream_length", "Entries in a result stream.", ["stream", "result_stream"], registry=self.registry
        )
        self._model_loaded_gauge.labels(stream).set(0)
        if manifest is not None:
            self._model_info.labels(stream, manifest.manifest_id, manifest.model).set(1)

    # ------------------------------------------------------------ recording

    @_safe
    def tick(self) -> None:
        """The main loop (or the lease heartbeat) made progress."""
        self._last_tick = self._clock()

    @_safe
    def redis_ok(self) -> None:
        """A Redis call succeeded."""
        self._last_redis_ok = self._clock()

    @_safe
    def message_settled(self, outcome: str) -> None:
        """A message ended with `outcome` (one of OUTCOMES)."""
        self._messages.labels(self.stream, outcome).inc()
        self._last_message.labels(self.stream).set(time.time())

    @_safe
    def batch_processed(self, size: int, seconds: float) -> None:
        self._batch_size.labels(self.stream).observe(size)
        self._duration.labels(self.stream).observe(seconds)

    @_safe
    def reclaimed(self, count: int) -> None:
        self._reclaimed.labels(self.stream).inc(count)

    @_safe
    def queue_snapshot(self, length: int, pending: int, dead_letters: int) -> None:
        self._queue_length.labels(self.stream).set(length)
        self._queue_pending.labels(self.stream).set(pending)
        self._dead_letter_length.labels(self.stream).set(dead_letters)

    @_safe
    def result_stream_length(self, result_stream: str, length: int) -> None:
        self._result_stream_length.labels(self.stream, result_stream).set(length)

    @_safe
    def model_loaded(self, loaded: bool) -> None:
        self._model_loaded = loaded
        self._model_loaded_gauge.labels(self.stream).set(1 if loaded else 0)

    # --------------------------------------------------------------- health

    def alive(self) -> bool:
        """The loop ticked recently."""
        return self._clock() - self._last_tick <= self._stale_after

    def ready(self) -> bool:
        """The loop ticked recently and Redis answered recently.

        The model is not part of readiness: models load on the first task, so an
        idle worker would never become ready. See ml_model_loaded.
        """
        return self.alive() and self._clock() - self._last_redis_ok <= self._stale_after

    def status(self) -> dict:
        now = self._clock()
        return {
            "stream": self.stream,
            "alive": self.alive(),
            "ready": self.ready(),
            "model_loaded": self._model_loaded,
            "seconds_since_tick": round(now - self._last_tick, 1),
            "seconds_since_redis_ok": round(now - self._last_redis_ok, 1),
        }

    def render_metrics(self) -> bytes:
        return generate_latest(self.registry)


class TelemetryServer:
    """Serves WorkerTelemetry on a background thread."""

    def __init__(self, telemetry: WorkerTelemetry, port: int, host: str = "0.0.0.0"):
        self._telemetry = telemetry
        self._address = (host, port)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> int:
        """Start serving. Returns the port (useful when port 0 asked for any free one)."""
        telemetry = self._telemetry

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 (name set by http.server)
                if self.path == "/metrics":
                    self._send(200, telemetry.render_metrics(), CONTENT_TYPE_LATEST)
                elif self.path in ("/healthz", "/readyz"):
                    ok = telemetry.alive() if self.path == "/healthz" else telemetry.ready()
                    body = json.dumps(telemetry.status()).encode("utf-8")
                    self._send(200 if ok else 503, body, "application/json")
                else:
                    self._send(404, b"not found", "text/plain")

            def _send(self, code: int, body: bytes, content_type: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):  # silence the per-request stderr line
                pass

        self._server = ThreadingHTTPServer(self._address, Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, name="telemetry", daemon=True)
        self._thread.start()
        return self._server.server_address[1]

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
