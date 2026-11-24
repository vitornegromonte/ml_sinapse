"""BaseWorker + telemetry: metrics, health, and log context around the same
ack/retry/dead-letter behavior as before (see test_base_worker.py)."""
import asyncio
import io
import json
import logging
import socket
import urllib.error
import urllib.request
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config import Settings
from services.logging_setup import build_handler
from services.model_manifest import StageManifest
from services.telemetry import WorkerTelemetry
from workers.base_worker import BaseWorker, PermanentTaskError


class _Worker(BaseWorker):
    """Fails or succeeds as told: `fail_with` is an exception or None."""

    def __init__(self, fail_with=None, cfg=None):
        super().__init__("s:test", cfg or Settings(_env_file=None))
        self.fail_with = fail_with

    async def process(self, data: dict) -> None:
        self.logger.info("working")
        if self.fail_with is not None:
            raise self.fail_with


class _BatchWorker(_Worker):
    supports_batch = True

    async def process_batch(self, items):
        self.logger.info("batch working")
        return [RuntimeError("one bad") if item.get("bad") else None for item in items]


def _redis():
    redis = AsyncMock()
    counts: dict = {}

    async def hincrby(key, field, amount):
        counts[field] = counts.get(field, 0) + amount
        return counts[field]

    redis.hincrby.side_effect = hincrby
    redis.exists = AsyncMock(return_value=0)
    return redis


def _value(telemetry, name, **labels):
    from prometheus_client.parser import text_string_to_metric_families

    for family in text_string_to_metric_families(telemetry.render_metrics().decode()):
        for sample in family.samples:
            if sample.name == name and labels.items() <= sample.labels.items():
                return sample.value
    return None


@pytest.fixture
def worker():
    w = _Worker()
    w.redis = _redis()
    w.telemetry = WorkerTelemetry("s:test")
    return w


class TestOutcomes:
    async def test_a_success_is_counted_and_still_acked(self, worker):
        """Arrange: a worker whose process() succeeds.
        Act: process one message.
        Assert: outcome success is 1, one batch of size 1 is observed, and the
        message was acked exactly as before.
        """
        await worker._process_message("1-0", {"x": 1})

        assert _value(worker.telemetry, "ml_messages_total", outcome="success") == 1.0
        assert _value(worker.telemetry, "ml_batch_size_count") == 1.0
        worker.redis.xack.assert_awaited_once()

    async def test_a_failure_with_retries_left_is_counted_as_retry_and_not_acked(self, worker):
        """Arrange: a failing worker with 3 retries allowed.
        Act: process one message.
        Assert: outcome retry is 1, and the message is NOT acked (so it can be reclaimed).
        """
        worker.fail_with = RuntimeError("boom")

        await worker._process_message("1-0", {"x": 1})

        assert _value(worker.telemetry, "ml_messages_total", outcome="retry") == 1.0
        worker.redis.xack.assert_not_awaited()

    async def test_the_last_failure_is_counted_as_dead_letter_and_acked(self):
        """Arrange: a failing worker with max_retries=1.
        Act: process one message.
        Assert: outcome dead_letter is 1, the payload went to the dead-letter stream, and it was acked.
        """
        worker = _Worker(RuntimeError("boom"), Settings(_env_file=None, max_retries=1))
        worker.redis, worker.telemetry = _redis(), WorkerTelemetry("s:test")

        await worker._process_message("1-0", {"x": 1})

        assert _value(worker.telemetry, "ml_messages_total", outcome="dead_letter") == 1.0
        assert worker.redis.xadd.call_args.args[0] == "s:test:dead"
        worker.redis.xack.assert_awaited_once()

    async def test_a_permanent_error_is_counted_as_permanent_and_dead_lettered_at_once(self, worker):
        """Arrange: a worker that raises PermanentTaskError.
        Act: process one message.
        Assert: outcome permanent is 1, it is dead-lettered and acked, and the retry counter is untouched.
        """
        worker.fail_with = PermanentTaskError("bad payload")

        await worker._process_message("1-0", {"x": 1})

        assert _value(worker.telemetry, "ml_messages_total", outcome="permanent") == 1.0
        assert worker.redis.xadd.call_args.args[0] == "s:test:dead"
        worker.redis.hincrby.assert_not_awaited()

    async def test_a_batch_counts_each_message_and_one_duration(self):
        """Arrange: a batch worker where one of three messages fails.
        Act: process the three together.
        Assert: 2 successes, 1 retry, one batch of size 3, and one duration observation.
        """
        worker = _BatchWorker()
        worker.redis, worker.telemetry = _redis(), WorkerTelemetry("s:test")

        await worker._process_messages([("1-0", {}), ("2-0", {"bad": True}), ("3-0", {})])

        assert _value(worker.telemetry, "ml_messages_total", outcome="success") == 2.0
        assert _value(worker.telemetry, "ml_messages_total", outcome="retry") == 1.0
        assert _value(worker.telemetry, "ml_batch_size_sum") == 3.0
        assert _value(worker.telemetry, "ml_process_duration_seconds_count") == 1.0

    async def test_without_telemetry_nothing_changes(self):
        """Arrange: a worker with telemetry off (the default).
        Act: process a success and a failure.
        Assert: telemetry is None, the success is acked and the failure is not.
        """
        ok, bad = _Worker(), _Worker(RuntimeError("boom"))
        ok.redis, bad.redis = _redis(), _redis()

        await ok._process_message("1-0", {})
        await bad._process_message("1-0", {})

        assert ok.telemetry is None and bad.telemetry is None
        ok.redis.xack.assert_awaited_once()
        bad.redis.xack.assert_not_awaited()

    async def test_a_broken_metric_cannot_change_the_outcome(self, worker, monkeypatch):
        """Arrange: a telemetry object whose counters raise.
        Act: process a success.
        Assert: the message is still acked, because observability never breaks processing.
        """
        def boom(*args, **kwargs):
            raise RuntimeError("registry broken")

        monkeypatch.setattr(worker.telemetry._messages, "labels", boom)
        monkeypatch.setattr(worker.telemetry._duration, "labels", boom)

        await worker._process_message("1-0", {})

        worker.redis.xack.assert_awaited_once()

    async def test_reclaimed_messages_are_counted(self, worker):
        """Arrange: XAUTOCLAIM hands back one stale message whose lease is gone.
        Act: run the reclaim.
        Assert: the reclaimed counter is 1, and the message was processed.
        """
        worker.redis.xautoclaim = AsyncMock(return_value=("0-0", [("1-0", {"x": "1"})], []))

        await worker._reclaim_stale()

        assert _value(worker.telemetry, "ml_messages_reclaimed_total") == 1.0
        worker.redis.xack.assert_awaited_once()

    async def test_the_lease_heartbeat_keeps_the_worker_alive(self):
        """Arrange: a worker with a very short heartbeat and a slow process().
        Act: process one message that takes 0.1 s.
        Assert: tick() was called by the heartbeat while the message ran.
        """
        class Slow(_Worker):
            async def process(self, data):
                await asyncio.sleep(0.1)

        worker = Slow(cfg=Settings(_env_file=None, lease_heartbeat_seconds=0))
        worker.redis, worker.telemetry = _redis(), WorkerTelemetry("s:test")
        worker.telemetry.tick = MagicMock()

        await worker._process_message("1-0", {})

        assert worker.telemetry.tick.call_count >= 1


def _json_logs(worker):
    buffer = io.StringIO()
    handler = build_handler("json", {"stream": "s:test"})
    handler.stream = buffer
    worker.logger.addHandler(handler)
    worker.logger.setLevel(logging.INFO)
    worker.logger.propagate = False
    return buffer, handler


class TestLogContext:
    async def test_a_single_message_logs_its_ids(self, worker):
        """Arrange: a worker that logs while processing, with a JSON handler.
        Act: process one message with a document id and a pipeline run id.
        Assert: the line written inside process() has the message id and both ids.
        """
        buffer, handler = _json_logs(worker)

        await worker._process_message("1700-0", {"document_id": "bo-1", "pipeline_run_id": "run-7", "text": "SECRET"})
        worker.logger.removeHandler(handler)

        working = next(json.loads(l) for l in buffer.getvalue().splitlines() if "working" in l)
        assert working["message_id"] == "1700-0"
        assert working["document_id"] == "bo-1" and working["pipeline_run_id"] == "run-7"
        assert "SECRET" not in buffer.getvalue()

    async def test_a_batch_logs_the_list_of_message_ids(self):
        """Arrange: a batch worker with a JSON handler.
        Act: process three messages together.
        Assert: the line has message_ids with all three, and no single message_id.
        """
        worker = _BatchWorker()
        worker.redis = _redis()
        buffer, handler = _json_logs(worker)

        await worker._process_messages([("1-0", {}), ("2-0", {}), ("3-0", {})])
        worker.logger.removeHandler(handler)

        line = next(json.loads(l) for l in buffer.getvalue().splitlines() if "batch working" in l)
        assert line["message_ids"] == ["1-0", "2-0", "3-0"]
        assert "message_id" not in line

    async def test_the_context_does_not_leak_after_the_message(self, worker):
        """Arrange: a worker and a JSON handler.
        Act: process a message, then log outside it.
        Assert: the later line has no message_id.
        """
        buffer, handler = _json_logs(worker)

        await worker._process_message("1-0", {"document_id": "bo-1"})
        worker.logger.info("later")
        worker.logger.removeHandler(handler)

        later = json.loads(buffer.getvalue().splitlines()[-1])
        assert later["message"] == "later" and "message_id" not in later


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


def _run_start(worker, during_first_poll=None):
    """Run worker.start() for one poll. `during_first_poll` is an async callable run inside it."""
    worker._connect_redis = AsyncMock(return_value=worker.redis)
    worker.redis.xautoclaim = AsyncMock(return_value=("0-0", [], []))
    worker.redis.xlen = AsyncMock(return_value=7)
    worker.redis.xpending = AsyncMock(return_value={"pending": 2})

    async def poll(*args, **kwargs):
        if during_first_poll is not None:
            await during_first_poll()
        worker._running = False
        return []

    worker.redis.xreadgroup = AsyncMock(side_effect=poll)
    engine = MagicMock()
    engine.dispose = AsyncMock()
    with patch("workers.base_worker.create_db_engine", return_value=engine), \
            patch("workers.base_worker.create_session_factory"):
        asyncio.run(worker.start())


class TestStart:
    def test_telemetry_is_off_unless_a_port_is_set(self):
        """Arrange: a worker with metrics_port 0 (the default).
        Act: run start() for one poll.
        Assert: no telemetry object and no server were created.
        """
        worker = _Worker()
        worker.redis = _redis()

        _run_start(worker)

        assert worker.telemetry is None and worker._telemetry_server is None

    def test_with_a_port_the_endpoints_answer_during_the_run_and_close_after(self):
        """Arrange: a worker with a port set, a manifest, and a queue holding 7 entries (2 pending).
        Act: during the first poll, GET /healthz, /readyz and /metrics; then let start() finish.
        Assert: 200, 200 and the expected metrics during the run, and the port is closed afterwards.
        """
        port = _free_port()
        worker = _Worker(cfg=Settings(_env_file=None, metrics_port=port, metrics_refresh_seconds=1))
        worker.redis = _redis()
        worker._manifest = StageManifest("ocr", "org/m@123456789abc", {}, "abc123def456")
        seen = {}

        async def during():
            await asyncio.sleep(0.2)  # let the first queue refresh run
            base = f"http://127.0.0.1:{port}"
            seen["health"] = await asyncio.to_thread(_get, base + "/healthz")
            seen["ready"] = await asyncio.to_thread(_get, base + "/readyz")
            seen["metrics"] = await asyncio.to_thread(_get, base + "/metrics")

        _run_start(worker, during)

        assert seen["health"][0] == 200 and seen["ready"][0] == 200
        metrics = seen["metrics"][1]
        assert 'ml_queue_length{stream="s:test"} 7.0' in metrics
        assert 'ml_queue_pending{stream="s:test"} 2.0' in metrics
        assert 'manifest_id="abc123def456"' in metrics
        with pytest.raises(urllib.error.URLError):
            urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2)

    def test_a_port_that_is_taken_does_not_stop_the_worker(self, caplog):
        """Arrange: a port already in use.
        Act: run start() with metrics_port set to it.
        Assert: start() finishes, an error is logged, and no server is held.
        """
        with socket.socket() as busy:
            busy.bind(("0.0.0.0", 0))
            busy.listen()
            worker = _Worker(cfg=Settings(_env_file=None, metrics_port=busy.getsockname()[1]))
            worker.redis = _redis()

            with caplog.at_level("ERROR"):
                _run_start(worker)

        assert "Could not start the metrics server" in caplog.text
        assert worker._telemetry_server is None


class TestQueueGauges:
    async def test_the_refresh_reads_length_pending_and_dead_letters(self, worker):
        """Arrange: a Redis with 7 entries, 2 pending, and 3 dead letters.
        Act: refresh the gauges.
        Assert: the three gauges show those numbers, and Redis counts as reachable.
        """
        worker.redis.xlen = AsyncMock(side_effect=lambda stream: 3 if stream.endswith(":dead") else 7)
        worker.redis.xpending = AsyncMock(return_value={"pending": 2})

        await worker._refresh_queue_gauges()

        assert _value(worker.telemetry, "ml_queue_length") == 7.0
        assert _value(worker.telemetry, "ml_queue_pending") == 2.0
        assert _value(worker.telemetry, "ml_dead_letter_length") == 3.0

    async def test_a_redis_fault_is_swallowed_and_leaves_the_gauges_alone(self, worker):
        """Arrange: a Redis whose XLEN fails, and gauges holding an earlier snapshot.
        Act: refresh.
        Assert: no exception, and the earlier values are unchanged.
        """
        worker.telemetry.queue_snapshot(5, 1, 0)
        worker.redis.xlen = AsyncMock(side_effect=ConnectionError("down"))

        await worker._refresh_queue_gauges()

        assert _value(worker.telemetry, "ml_queue_length") == 5.0


def _model_loaded(telemetry):
    return _value(telemetry, "ml_model_loaded")


class TestModelLoadedGauge:
    def test_ocr_worker_sets_and_clears_the_gauge(self):
        """Arrange: an OCR worker with telemetry and create_model patched.
        Act: load, then unload the model.
        Assert: ml_model_loaded is 1 after loading and 0 after unloading.
        """
        from workers.ocr_worker import OcrWorker

        worker = OcrWorker(Settings(_env_file=None))
        worker.telemetry = WorkerTelemetry(worker.stream)
        with patch("workers.ocr_worker.create_model"):
            worker._load_model()
            loaded = _model_loaded(worker.telemetry)
            worker._unload_model()

        assert (loaded, _model_loaded(worker.telemetry)) == (1.0, 0.0)

    def test_llm_worker_sets_and_clears_the_gauge(self):
        """Arrange: an LLM worker with telemetry and LLMModel patched.
        Act: load, then unload.
        Assert: 1, then 0.
        """
        from workers.llm_worker import LlmWorker

        worker = LlmWorker(Settings(_env_file=None))
        worker.telemetry = WorkerTelemetry(worker.stream)
        with patch("workers.llm_worker.LLMModel"):
            worker._load_model()
            loaded = _model_loaded(worker.telemetry)
            worker._unload_model()

        assert (loaded, _model_loaded(worker.telemetry)) == (1.0, 0.0)

    def test_embedding_worker_sets_the_gauge(self):
        """Arrange: an embedding worker with telemetry and the encoder patched.
        Act: load the model.
        Assert: the gauge is 1.
        """
        from workers.embedding_worker import EmbeddingWorker

        worker = EmbeddingWorker(Settings(_env_file=None))
        worker.telemetry = WorkerTelemetry(worker.stream)
        with patch("workers.embedding_worker.SentenceTransformerEncoder"):
            worker._load_model()

        assert _model_loaded(worker.telemetry) == 1.0

    def test_linkage_worker_sets_the_gauge_only_after_the_classifier_checks(self):
        """Arrange: a linkage worker with the encoder and classifier loading patched.
        Act: load with a classifier of 7 features, then with one of 3.
        Assert: the gauge is 1 for the good one, and the bad one raises before the gauge can be set.
        """
        from services.model_compat import ModelCompatibilityError
        from workers.linkage_scoring_worker import LinkageScoringWorker

        good = LinkageScoringWorker(Settings(_env_file=None))
        good.telemetry = WorkerTelemetry(good.stream)
        with patch("workers.linkage_scoring_worker.SentenceTransformerEncoder"), \
                patch("joblib.load", return_value=MagicMock(n_features_in_=7)):
            good._load_model()

        bad = LinkageScoringWorker(Settings(_env_file=None))
        bad.telemetry = WorkerTelemetry(bad.stream)
        with patch("workers.linkage_scoring_worker.SentenceTransformerEncoder"), \
                patch("joblib.load", return_value=MagicMock(n_features_in_=3)), \
                pytest.raises(ModelCompatibilityError):
            bad._load_model()

        assert _model_loaded(good.telemetry) == 1.0
        assert _model_loaded(bad.telemetry) == 0.0

    def test_ner_worker_sets_the_gauge(self):
        """Arrange: a NER worker with telemetry and spaCy loading patched.
        Act: load the model.
        Assert: the gauge is 1.
        """
        from workers.ner_worker import NerWorker

        worker = NerWorker(Settings(_env_file=None))
        worker.telemetry = WorkerTelemetry(worker.stream)
        with patch("workers.ner_worker.sync_if_missing"), patch("workers.ner_worker.spacy.load"):
            worker._load_model()

        assert _model_loaded(worker.telemetry) == 1.0

    def test_without_telemetry_loading_still_works(self):
        """Arrange: an OCR worker with telemetry off.
        Act: load the model.
        Assert: no error.
        """
        from workers.ocr_worker import OcrWorker

        with patch("workers.ocr_worker.create_model"):
            OcrWorker(Settings(_env_file=None))._load_model()


class TestResultStreamGauge:
    async def test_the_ocr_publisher_reports_the_result_stream_length(self, mock_redis):
        """Arrange: an OCR worker with telemetry, and a Redis whose result stream has 5 entries.
        Act: publish one result through the worker's publisher.
        Assert: ml_result_stream_length for ocr:tasks:results is 5.
        """
        from workers.ocr_worker import OcrWorker

        worker = OcrWorker(Settings(_env_file=None))
        worker.redis, worker.telemetry = mock_redis, WorkerTelemetry(worker.stream)
        mock_redis.xlen.return_value = 5

        worker._ensure_publisher()
        await worker._result_publisher.publish({"document_id": "d1"}, "texto")

        assert _value(worker.telemetry, "ml_result_stream_length", result_stream="ocr:tasks:results") == 5.0

    async def test_the_scoring_worker_reports_its_result_stream_length(self, mock_redis):
        """Arrange: a scoring worker with telemetry, and a result stream of 9 entries.
        Act: process one task.
        Assert: the gauge for scoring:tasks:results is 9.
        """
        from workers.scoring_worker import ScoringWorker

        worker = ScoringWorker(Settings(_env_file=None))
        worker.redis, worker.telemetry = mock_redis, WorkerTelemetry(worker.stream)
        mock_redis.xlen.return_value = 9

        await worker.process({"boletim_id": 1, "ato_infracional": "HOMICIDIO", "features": {"victim": True}})

        assert _value(worker.telemetry, "ml_result_stream_length", result_stream="scoring:tasks:results") == 9.0
