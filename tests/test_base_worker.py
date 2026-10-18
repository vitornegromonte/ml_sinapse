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
        """A message that fails but hasn't hit max_retries must stay pending
        (never acked) — an acked Streams message can never be redelivered."""
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
        """Once max_retries is hit, the message IS dead-lettered and acked
        (removed from the pending list) — this path is unchanged."""
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
        """Locks in the real redis-py xautoclaim signature (start_id, not
        start) and confirms the sweep loops until the cursor wraps to 0-0."""
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
        """A Redis error during reclaim shouldn't crash the worker's main loop."""
        w = _FailNTimesWorker(fail_times=0, stream="test:stream")
        w.redis = AsyncMock()
        w.redis.xautoclaim.side_effect = RuntimeError("connection lost")

        await w._reclaim_stale()  # must not raise

    async def test_does_not_reclaim_a_message_whose_lease_is_alive(self):
        """A message another replica is still processing (heartbeat lease
        alive) must not be stolen — that duplication was the bug."""
        w = _FailNTimesWorker(fail_times=0, stream="test:stream")
        w.redis = AsyncMock()
        w.redis.exists = AsyncMock(return_value=1)  # lease still held
        w.redis.xautoclaim.side_effect = [
            ("0-0", [("busy-msg", {"x": 1})], []),
        ]

        await w._reclaim_stale()

        assert w.attempts == 0  # skipped, not reprocessed
