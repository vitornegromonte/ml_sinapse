"""Tests for the task producer's memory-bounding helpers.

No real Redis: the module-level client is monkeypatched. The point is the
backpressure *logic* — lag (not XLEN) is the backlog signal, and an unknown
lag must not stall the producer.
"""

from unittest.mock import AsyncMock

from redis.exceptions import ResponseError

import services.task_publisher as tp


def _client(groups=None, error=None):
    c = AsyncMock()
    if error is not None:
        c.xinfo_groups = AsyncMock(side_effect=error)
    else:
        c.xinfo_groups = AsyncMock(return_value=groups or [])
    return c


class TestBacklog:
    async def test_reads_group_lag(self, monkeypatch):
        """Arrange: a Redis client reporting a consumer-group lag of 42.
        Act: read the backlog.
        Assert: 42 is returned — lag, not XLEN, is the backlog signal.
        """
        monkeypatch.setattr(tp, "_client", _client([{"name": "ml-workers", "lag": 42}]))
        assert await tp.backlog("embedding:tasks") == 42

    async def test_none_when_server_cannot_report_lag(self, monkeypatch):
        """Arrange: a consumer group whose lag is unknown.
        Act: read the backlog.
        Assert: None, so the caller can tell "unknown" from "empty".
        """
        monkeypatch.setattr(tp, "_client", _client([{"name": "ml-workers", "lag": None}]))
        assert await tp.backlog("embedding:tasks") is None

    async def test_zero_when_group_absent(self, monkeypatch):
        """Arrange: a stream with no consumer group yet.
        Act: read the backlog.
        Assert: 0 — nothing has been consumed, but nothing is stuck either.
        """
        monkeypatch.setattr(tp, "_client", _client([]))
        assert await tp.backlog("embedding:tasks") == 0

    async def test_zero_when_stream_absent(self, monkeypatch):
        """Arrange: a Redis client that raises "no such key".
        Act: read the backlog.
        Assert: 0, because a missing stream has no pending work.
        """
        monkeypatch.setattr(tp, "_client", _client(error=ResponseError("no such key")))
        assert await tp.backlog("embedding:tasks") == 0


class TestWaitForCapacity:
    async def test_returns_immediately_below_limit(self, monkeypatch):
        """Arrange: a lag of 5 with a pending limit of 100.
        Act: wait for capacity.
        Assert: it returns without sleeping.
        """
        monkeypatch.setattr(tp, "_client", _client([{"name": "ml-workers", "lag": 5}]))
        await tp.wait_for_capacity("embedding:tasks", pending_limit=100)

    async def test_does_not_stall_when_lag_unknown(self, monkeypatch):
        """Arrange: a lag the server cannot report.
        Act: wait for capacity.
        Assert: it returns immediately — an unknown lag must not stall the
        producer.
        """
        monkeypatch.setattr(tp, "_client", _client([{"name": "ml-workers", "lag": None}]))
        await tp.wait_for_capacity("embedding:tasks", pending_limit=100)

    async def test_blocks_until_backlog_drains(self, monkeypatch):
        """Arrange: a lag sequence of 3 then 1 with a pending limit of 2, and
        sleep patched to record its calls.
        Act: wait for capacity.
        Assert: it sleeps once for 1 s, then returns when the lag drops below
        the limit.
        """
        lags = [3, 1]
        c = AsyncMock()

        async def xinfo_groups(_stream):
            return [{"name": "ml-workers", "lag": lags.pop(0)}]

        c.xinfo_groups = xinfo_groups
        monkeypatch.setattr(tp, "_client", c)

        slept = []

        async def fake_sleep(seconds):
            slept.append(seconds)

        monkeypatch.setattr(tp.asyncio, "sleep", fake_sleep)

        await tp.wait_for_capacity("embedding:tasks", pending_limit=2)
        assert slept == [1.0]
