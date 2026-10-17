"""Redis Streams producer for pipeline task messages (backfill/trigger scripts).

Mirrors the backend's producer (`app/core/queue.py.enqueue` in the sinapse
repo): field values are stringified (Redis Streams fields must be strings)
and the generated message id is returned. Stream names come from Settings —
they already mirror the backend's stream constants by convention.
"""

from typing import Any, Mapping

import redis.asyncio as redis

from config import settings


async def enqueue(stream: str, payload: Mapping[str, Any]) -> str:
    """Add a task to a Redis stream. Returns the generated message id."""
    client = redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_connect_timeout=10,
    )
    try:
        fields = {k: str(v) for k, v in payload.items()}
        return await client.xadd(stream, fields)
    finally:
        await client.aclose()
