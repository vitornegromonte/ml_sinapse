"""Regression tests for BaseWorker's retry/reclaim mechanics.

These were previously only verified via ad-hoc scripts during development
(never committed) — which is exactly how a real bug shipped anyway: an
AsyncMock accepts any keyword argument name, so a wrong kwarg to a real
redis-py method (xautoclaim's `start` vs. the actual `start_id`) passed
every mocked test here and only broke against a real Redis server. These
tests can't catch THAT class of bug either (same mocking limitation) — they
exist to lock in the retry/ack *logic*, not the exact redis-py call shape.
"""

from unittest.mock import AsyncMock

import pytest

from workers.base_worker import BaseWorker


class _FailNTimesWorker(BaseWorker):
    def __init__(self, fail_times: int, *a, **kw):
        super().__init__(*a, **kw)
        self.attempts = 0
        self._fail_times = fail_times

    async def process(self, data: dict) -> None:
        self.attempts += 1
        if self.attempts <= self._fail_times:
            raise RuntimeError("boom")


def _make_redis_with_retry_counter():
    redis = AsyncMock()
    counts: dict[str, int] = {}

    async def hincrby(key, field, amt):
        counts[field] = counts.get(field, 0) + amt
        return counts[field]

    async def hget(key, field):
        return counts.get(field)

    redis.hincrby.side_effect = hincrby
    redis.hget.side_effect = hget
    return redis, counts


class TestProcessMessageRetry:
    async def test_failed_message_is_not_acked_before_exhausting_retries(self):
        """Arrange: a worker failing the first two attempts, max_retries=3.
        Act: process the same message three times.
        Assert: it is never acked before the successful third attempt — an acked
        Streams message can never be redelivered.
        """
        w = _FailNTimesWorker(fail_times=2, stream="test:stream")
        w.cfg.max_retries = 3
        w.redis, _ = _make_redis_with_retry_counter()

        await w._process_message("msg-1", {"x": 1})
        assert w.redis.xack.call_count == 0

        await w._process_message("msg-1", {"x": 1})  # simulates XAUTOCLAIM redelivery
        assert w.redis.xack.call_count == 0

        await w._process_message("msg-1", {"x": 1})  # succeeds on 3rd attempt
        assert w.redis.xack.call_count == 1
        assert w.attempts == 3

    async def test_exhausted_retries_dead_letters_and_acks(self):
        """Arrange: a worker that always fails, max_retries=2.
        Act: process the same message twice.
        Assert: on the second attempt it is written to the DLQ and acked,
        removing it from the pending list.
        """
        w = _FailNTimesWorker(fail_times=99, stream="test:stream")
        w.cfg.max_retries = 2
        w.redis, _ = _make_redis_with_retry_counter()

        await w._process_message("msg-x", {"x": 1})  # retries=1 < 2
        assert w.redis.xack.call_count == 0

        await w._process_message("msg-x", {"x": 1})  # retries=2 >= 2
        assert w.redis.xadd.call_count == 1, "should have written to the DLQ stream"
        assert w.redis.xack.call_count == 1, "should ack after dead-lettering"


class TestReclaimStale:
    async def test_calls_xautoclaim_with_correct_kwargs_and_loops_to_completion(self):
        """Arrange: a Redis client whose xautoclaim returns one message and then
        the 0-0 cursor, with no live lease.
        Act: reclaim stale messages.
        Assert: two calls with min_idle_time and the real start_id keyword (not
        start), and the claimed message is processed.
        """
        w = _FailNTimesWorker(fail_times=99, stream="test:stream")
        w.cfg.claim_idle_ms = 30000
        w.cfg.batch_size = 1
        w.redis = AsyncMock()
        # redis-py's hincrby returns int; a bare AsyncMock would return a
        # mock and break the `retries >= max_retries` comparison in
        # _process_message (pre-existing env-drift failure).
        w.redis.hincrby = AsyncMock(return_value=1)
        w.redis.xautoclaim.side_effect = [
            ("123-0", [("claimed-msg-1", {"x": 1})], []),
            ("0-0", [], []),
        ]
        w.redis.exists = AsyncMock(return_value=0)  # no live lease

        await w._reclaim_stale()

        assert w.redis.xautoclaim.call_count == 2
        first_call_kwargs = w.redis.xautoclaim.call_args_list[0].kwargs
        assert first_call_kwargs["min_idle_time"] == 30000
        assert first_call_kwargs["start_id"] == "0-0"
        assert "start" not in first_call_kwargs  # the wrong name this test would have caught
        assert w.attempts == 1  # the claimed message was fed through _process_message

    async def test_reclaim_errors_are_caught_not_raised(self):
        """Arrange: a Redis client whose xautoclaim raises a connection error.
        Act: reclaim stale messages.
        Assert: it does not raise, so the worker's main loop keeps running.
        """
        w = _FailNTimesWorker(fail_times=0, stream="test:stream")
        w.redis = AsyncMock()
        w.redis.xautoclaim.side_effect = RuntimeError("connection lost")

        await w._reclaim_stale()  # must not raise

    async def test_does_not_reclaim_a_message_whose_lease_is_alive(self):
        """Arrange: xautoclaim returns a message whose heartbeat lease exists.
        Act: reclaim stale messages.
        Assert: it is not processed — another replica still owns it, and that
        duplication was the bug.
        """
        w = _FailNTimesWorker(fail_times=0, stream="test:stream")
        w.redis = AsyncMock()
        w.redis.exists = AsyncMock(return_value=1)  # lease still held
        w.redis.xautoclaim.side_effect = [
            ("0-0", [("busy-msg", {"x": 1})], []),
        ]

        await w._reclaim_stale()

        assert w.attempts == 0  # skipped, not reprocessed


class _BatchWorker(BaseWorker):
    """Worker que manda o lote inteiro numa chamada, como embedding e llm."""

    supports_batch = True

    def __init__(self, fail_items=(), *a, **kw):
        super().__init__(*a, **kw)
        self.calls: list[list[dict]] = []
        self._fail_items = set(fail_items)

    async def process(self, data: dict) -> None:
        raise AssertionError("process_batch deve ser usado quando supports_batch")

    async def process_batch(self, items: list[dict]) -> list[Exception | None]:
        self.calls.append(items)
        return [
            RuntimeError("bad") if data["x"] in self._fail_items else None
            for data in items
        ]


class TestProcessMessagesBatch:
    async def test_one_batch_call_and_one_ack_per_message(self):
        """Arrange: a batch worker and two messages.
        Act: process the messages.
        Assert: one process_batch call with both items, one ack per message, and
        one lease set/deleted per message.
        """
        w = _BatchWorker(stream="test:stream")
        w.cfg.max_retries = 3
        w.redis, _ = _make_redis_with_retry_counter()

        await w._process_messages([("m1", {"x": 1}), ("m2", {"x": 2})])

        assert w.calls == [[{"x": 1}, {"x": 2}]]
        assert [c.args[2] for c in w.redis.xack.call_args_list] == ["m1", "m2"]
        # Um lease por mensagem, liberado no fim.
        assert w.redis.set.call_count == 2
        assert w.redis.delete.call_count == 2

    async def test_failed_item_is_settled_alone_after_the_batch_call(self):
        """Arrange: a batch worker where item 2 fails and items 1 and 3 succeed.
        Act: process all three.
        Assert: only 1 and 3 are acked and item 2 alone gets its retry counter
        incremented.
        """
        w = _BatchWorker(fail_items={2}, stream="test:stream")
        w.cfg.max_retries = 3
        w.redis, _ = _make_redis_with_retry_counter()

        await w._process_messages(
            [("m1", {"x": 1}), ("m2", {"x": 2}), ("m3", {"x": 3})]
        )

        assert [c.args[2] for c in w.redis.xack.call_args_list] == ["m1", "m3"]
        assert w.redis.hincrby.call_args_list[0].args == (
            "test:stream:retries",
            "m2",
            1,
        )

    async def test_batch_level_error_fails_every_item(self):
        """Arrange: a batch worker whose process_batch raises CUDA OOM.
        Act: process two messages.
        Assert: both retry counters increment and neither is acked — the error
        says nothing about each item.
        """
        w = _BatchWorker(stream="test:stream")
        w.cfg.max_retries = 3
        w.redis, _ = _make_redis_with_retry_counter()

        async def boom(items):
            raise RuntimeError("cuda oom")

        w.process_batch = boom

        await w._process_messages([("m1", {"x": 1}), ("m2", {"x": 2})])

        assert w.redis.hincrby.call_count == 2
        assert w.redis.xack.call_count == 0

    async def test_default_hook_runs_process_per_item(self):
        """Arrange: a serial worker (no batch support) and two messages.
        Act: process the messages.
        Assert: process runs once per item and both are acked.
        """
        w = _FailNTimesWorker(fail_times=0, stream="test:stream")
        w.redis, _ = _make_redis_with_retry_counter()

        await w._process_messages([("m1", {"x": 1}), ("m2", {"x": 2})])

        assert w.attempts == 2
        assert w.redis.xack.call_count == 2


class TestBatchCount:
    async def test_serial_worker_reads_one_message_even_with_batch_size(self):
        """Arrange: a serial worker with batch_size=8 configured.
        Act: ask how many messages to read.
        Assert: one — serial workers ignore batch_size.
        """
        w = _FailNTimesWorker(fail_times=0, stream="test:stream")
        w.cfg.batch_size = 8
        assert w._batch_count() == 1

    async def test_batching_worker_reads_batch_size(self):
        """Arrange: a batch worker with batch_size=8 configured.
        Act: ask how many messages to read.
        Assert: eight — batch workers read the full batch.
        """
        w = _BatchWorker(stream="test:stream")
        w.cfg.batch_size = 8
        assert w._batch_count() == 8
