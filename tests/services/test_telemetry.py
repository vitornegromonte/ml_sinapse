import json
import urllib.error
import urllib.request

import pytest
from prometheus_client.parser import text_string_to_metric_families

from services.model_manifest import StageManifest
from services.telemetry import OUTCOMES, TelemetryServer, WorkerTelemetry


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return _Clock()


@pytest.fixture
def telemetry(clock):
    return WorkerTelemetry("ocr:tasks", stale_after_seconds=80, clock=clock)


def _all_samples(telemetry):
    """Every sample of the rendered metrics: (sample name, labels dict, value)."""
    return [
        (sample.name, dict(sample.labels), sample.value)
        for family in text_string_to_metric_families(telemetry.render_metrics().decode())
        for sample in family.samples
    ]


def _value(telemetry, name, **labels):
    """The value of the sample called `name` whose labels include `labels`, or None."""
    for sample_name, sample_labels, value in _all_samples(telemetry):
        if sample_name == name and labels.items() <= sample_labels.items():
            return value
    return None


class TestMetrics:
    @pytest.mark.parametrize("outcome", OUTCOMES)
    def test_each_outcome_is_counted_under_its_own_label(self, telemetry, outcome):
        """Arrange: a telemetry object.
        Act: settle two messages with one outcome.
        Assert: ml_messages_total{outcome} is 2.
        """
        telemetry.message_settled(outcome)
        telemetry.message_settled(outcome)

        assert _value(telemetry, "ml_messages_total", outcome=outcome) == 2.0

    def test_a_batch_is_one_duration_observation_and_one_size_observation(self, telemetry):
        """Arrange: a telemetry object.
        Act: record one batch of 4 messages that took 2.5 seconds.
        Assert: the duration histogram has one observation summing to 2.5 and
        the batch-size histogram has one observation summing to 4.
        """
        telemetry.batch_processed(4, 2.5)

        assert _value(telemetry, "ml_process_duration_seconds_count") == 1.0
        assert _value(telemetry, "ml_process_duration_seconds_sum") == 2.5
        assert _value(telemetry, "ml_batch_size_sum") == 4.0

    def test_queue_gauges_hold_the_last_snapshot(self, telemetry):
        """Arrange: a telemetry object.
        Act: record two queue snapshots.
        Assert: the gauges show the second one.
        """
        telemetry.queue_snapshot(10, 3, 1)
        telemetry.queue_snapshot(7, 2, 4)

        assert _value(telemetry, "ml_queue_length") == 7.0
        assert _value(telemetry, "ml_queue_pending") == 2.0
        assert _value(telemetry, "ml_dead_letter_length") == 4.0

    def test_reclaimed_messages_are_counted(self, telemetry):
        """Arrange: a telemetry object.
        Act: record 2 and then 3 reclaimed messages.
        Assert: the counter is 5.
        """
        telemetry.reclaimed(2)
        telemetry.reclaimed(3)

        assert _value(telemetry, "ml_messages_reclaimed_total") == 5.0

    def test_model_loaded_starts_at_zero_and_follows_the_calls(self, telemetry):
        """Arrange: a new telemetry object.
        Act: read the gauge, load, then unload.
        Assert: 0, then 1, then 0.
        """
        before = _value(telemetry, "ml_model_loaded")
        telemetry.model_loaded(True)
        loaded = _value(telemetry, "ml_model_loaded")
        telemetry.model_loaded(False)

        assert (before, loaded, _value(telemetry, "ml_model_loaded")) == (0.0, 1.0, 0.0)

    def test_the_model_recipe_is_an_info_metric(self, clock):
        """Arrange: a manifest with an id and a model label.
        Act: build the telemetry object with it.
        Assert: ml_model_info carries both as labels, with value 1.
        """
        manifest = StageManifest("llm", "org/m@123456789abc", {}, "abc123def456")

        telemetry = WorkerTelemetry("llm:tasks", manifest, clock=clock)

        assert _value(
            telemetry, "ml_model_info",
            stream="llm:tasks", manifest_id="abc123def456", model="org/m@123456789abc",
        ) == 1.0

    def test_the_result_stream_length_is_labeled_by_stream(self, telemetry):
        """Arrange: a telemetry object.
        Act: record the length of a result stream.
        Assert: the gauge has the result stream as a label.
        """
        telemetry.result_stream_length("ocr:tasks:results", 1234)

        assert _value(telemetry, "ml_result_stream_length", result_stream="ocr:tasks:results") == 1234.0

    def test_the_metrics_hold_no_document_ids(self, telemetry):
        """Arrange: a telemetry object that recorded messages.
        Act: render the metrics.
        Assert: no label named document_id, boletim_id, pipeline_run_id or text
        appears, so the endpoint exposes no document data.
        """
        telemetry.message_settled("success")
        telemetry.batch_processed(1, 0.1)
        text = telemetry.render_metrics().decode()

        for forbidden in ("document_id", "boletim_id", "pipeline_run_id", "text="):
            assert forbidden not in text

    def test_a_failing_metric_never_raises(self, telemetry, monkeypatch):
        """Arrange: a metric whose labels() call raises.
        Act: record a settled message, a batch, and a queue snapshot.
        Assert: nothing raises, since observability must not break processing.
        """
        def boom(*args, **kwargs):
            raise RuntimeError("registry broken")

        monkeypatch.setattr(telemetry._messages, "labels", boom)
        monkeypatch.setattr(telemetry._duration, "labels", boom)
        monkeypatch.setattr(telemetry._queue_length, "labels", boom)

        telemetry.message_settled("success")
        telemetry.batch_processed(1, 0.1)
        telemetry.queue_snapshot(1, 1, 1)


class TestHealth:
    def test_a_fresh_worker_is_alive_and_ready(self, telemetry):
        """Arrange: a telemetry object just created.
        Act: ask for alive and ready.
        Assert: both true.
        """
        assert telemetry.alive() and telemetry.ready()

    def test_a_silent_loop_goes_stale(self, telemetry, clock):
        """Arrange: a telemetry object with an 80 second limit.
        Act: advance the clock by 81 seconds without a tick.
        Assert: not alive and not ready.
        """
        clock.now += 81

        assert not telemetry.alive() and not telemetry.ready()

    def test_a_tick_makes_the_worker_alive_again(self, telemetry, clock):
        """Arrange: a stalled worker.
        Act: tick.
        Assert: alive again (Redis was also fine, so ready too after redis_ok).
        """
        clock.now += 81
        telemetry.tick()
        telemetry.redis_ok()

        assert telemetry.alive() and telemetry.ready()

    def test_a_worker_that_lost_redis_is_alive_but_not_ready(self, telemetry, clock):
        """Arrange: a loop that keeps ticking while Redis stops answering.
        Act: advance the clock, tick, but record no Redis success.
        Assert: alive is true and ready is false.
        """
        clock.now += 81
        telemetry.tick()

        assert telemetry.alive() and not telemetry.ready()

    def test_the_boundary_is_inclusive(self, telemetry, clock):
        """Arrange: a telemetry object with an 80 second limit.
        Act: advance exactly 80 seconds.
        Assert: still alive.
        """
        clock.now += 80

        assert telemetry.alive()

    def test_readiness_does_not_wait_for_the_model(self, telemetry):
        """Arrange: a worker whose model is not loaded (models load on the first task).
        Act: ask for ready.
        Assert: true, and the status says model_loaded is false.
        """
        assert telemetry.ready()
        assert telemetry.status()["model_loaded"] is False

    def test_status_reports_the_ages(self, telemetry, clock):
        """Arrange: a telemetry object, with 12.5 seconds passed.
        Act: read the status.
        Assert: the ages and the stream are in it.
        """
        clock.now += 12.5

        status = telemetry.status()

        assert status["seconds_since_tick"] == 12.5
        assert status["stream"] == "ocr:tasks"


@pytest.fixture
def served(telemetry):
    server = TelemetryServer(telemetry, 0, host="127.0.0.1")
    port = server.start()
    yield telemetry, f"http://127.0.0.1:{port}", server
    server.stop()


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            return response.status, response.read().decode(), response.headers
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode(), error.headers


class TestServer:
    def test_metrics_are_served_in_prometheus_format(self, served):
        """Arrange: a running server.
        Act: GET /metrics after recording a message.
        Assert: 200, the Prometheus content type, and the counter's line.
        """
        telemetry, base, _ = served
        telemetry.message_settled("success")

        status, body, headers = _get(base + "/metrics")

        assert status == 200
        assert headers["Content-Type"].startswith("text/plain")
        assert 'ml_messages_total{outcome="success",stream="ocr:tasks"} 1.0' in body

    def test_healthz_is_200_then_503_when_the_loop_stalls(self, served, clock):
        """Arrange: a running server and a fake clock.
        Act: GET /healthz, advance 81 seconds, GET again.
        Assert: 200 first, then 503 with alive false in the JSON body.
        """
        _, base, _ = served

        first = _get(base + "/healthz")
        clock.now += 81
        second = _get(base + "/healthz")

        assert first[0] == 200
        assert second[0] == 503 and json.loads(second[1])["alive"] is False

    def test_readyz_is_503_when_redis_is_lost_but_healthz_stays_200(self, served, clock):
        """Arrange: a running server and a loop that ticks while Redis is silent.
        Act: advance 81 seconds, tick, GET both paths.
        Assert: /healthz 200 and /readyz 503.
        """
        telemetry, base, _ = served
        clock.now += 81
        telemetry.tick()

        assert _get(base + "/healthz")[0] == 200
        assert _get(base + "/readyz")[0] == 503

    def test_an_unknown_path_is_404(self, served):
        """Arrange: a running server.
        Act: GET /admin.
        Assert: 404.
        """
        assert _get(served[1] + "/admin")[0] == 404

    def test_stop_releases_the_port(self, telemetry):
        """Arrange: a started server.
        Act: stop it.
        Assert: a new request is refused.
        """
        server = TelemetryServer(telemetry, 0, host="127.0.0.1")
        port = server.start()
        server.stop()

        with pytest.raises(urllib.error.URLError):
            urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2)

    def test_stop_is_safe_to_call_twice(self, telemetry):
        """Arrange: a stopped server.
        Act: stop it again.
        Assert: no error.
        """
        server = TelemetryServer(telemetry, 0, host="127.0.0.1")
        server.start()
        server.stop()

        server.stop()
