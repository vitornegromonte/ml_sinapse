"""Redis Streams producer for pipeline task messages (backfill/trigger scripts).

Mirrors the backend's producer (`app/core/queue.py.enqueue` in the sinapse
repo): field values are stringified (Redis Streams fields must be strings)
and the generated message id is returned. Stream names come from Settings —
they already mirror the backend's stream constants by convention.

The client is created once and reused: the backfill scripts enqueue tens of
thousands of messages, and opening a TCP connection per message was pure
overhead (and a `maxclients` risk).
"""

import asyncio
import logging
from typing import Any, Mapping

import redis.asyncio as redis
from redis.exceptions import ResponseError

from config import settings

logger = logging.getLogger(__name__)

# Block the producer while the worker group is this many messages behind, so
# the whole backlog can't sit in Redis RAM. The task streams are never
# trimmed, so this is a real cap on memory, not a message-drop policy.
DEFAULT_PENDING_LIMIT = 10_000

_client: redis.Redis | None = None


def _get_client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=10,
        )
    return _client


async def enqueue(stream: str, payload: Mapping[str, Any]) -> str:
    """Add a task to a Redis stream. Returns the generated message id."""
    fields = {k: str(v) for k, v in payload.items()}
    return await _get_client().xadd(stream, fields)


async def backlog(stream: str) -> int | None:
    """Messages added but not yet read by the worker group.

    Uses the group's `lag` (Redis >= 7). This is NOT XLEN: the task streams
    are never trimmed, so XLEN only ever grows and would block the producer
    forever. Returns None when the server can't report a lag (pre-7) — callers
    must skip throttling then, not treat it as zero backlog.
    """
    try:
        groups = await _get_client().xinfo_groups(stream)
    except ResponseError:
        return 0  # stream (or group) doesn't exist yet: nothing queued
    for g in groups:
        if g.get("name") == settings.consumer_group:
            lag = g.get("lag")
            return int(lag) if lag is not None else None
    return 0


async def wait_for_capacity(
    stream: str,
    pending_limit: int = DEFAULT_PENDING_LIMIT,
    poll_seconds: float = 1.0,
) -> None:
    """Block while the worker group is `pending_limit` messages behind."""
    warned = False
    while True:
        lag = await backlog(stream)
        if lag is None:
            if not warned:
                logger.warning(
                    "Consumer lag unavailable for %s (Redis < 7?) — producer "
                    "backpressure is off and Redis memory is not bounded.",
                    stream,
                )
                warned = True
            return
        if lag < pending_limit:
            return
        await asyncio.sleep(poll_seconds)
