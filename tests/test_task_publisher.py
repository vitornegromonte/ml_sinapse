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
        monkeypatch.setattr(tp, "_client", _client([{"name": "ml-workers", "lag": 42}]))
        assert await tp.backlog("embedding:tasks") == 42

    async def test_none_when_server_cannot_report_lag(self, monkeypatch):
        monkeypatch.setattr(tp, "_client", _client([{"name": "ml-workers", "lag": None}]))
        assert await tp.backlog("embedding:tasks") is None

    async def test_zero_when_group_absent(self, monkeypatch):
        monkeypatch.setattr(tp, "_client", _client([]))
        assert await tp.backlog("embedding:tasks") == 0

    async def test_zero_when_stream_absent(self, monkeypatch):
        monkeypatch.setattr(tp, "_client", _client(error=ResponseError("no such key")))
        assert await tp.backlog("embedding:tasks") == 0


class TestWaitForCapacity:
    async def test_returns_immediately_below_limit(self, monkeypatch):
        monkeypatch.setattr(tp, "_client", _client([{"name": "ml-workers", "lag": 5}]))
        await tp.wait_for_capacity("embedding:tasks", pending_limit=100)

    async def test_does_not_stall_when_lag_unknown(self, monkeypatch):
        monkeypatch.setattr(tp, "_client", _client([{"name": "ml-workers", "lag": None}]))
        await tp.wait_for_capacity("embedding:tasks", pending_limit=100)

    async def test_blocks_until_backlog_drains(self, monkeypatch):
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
