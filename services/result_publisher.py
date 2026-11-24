import logging

from redis.asyncio import Redis as AsyncRedis

logger = logging.getLogger(__name__)


class ResultPublisher:
    """Publishes a worker's result to `<stream>:results`.

    `lineage` is a dict of extra fields (manifest_id, model) added to every
    message, so the consumer can tell which model recipe wrote it. Fields are
    only ever added: the consumer reads the ones it knows (ADR-014).
    """

    def __init__(
        self,
        redis: AsyncRedis,
        stream: str,
        maxlen: int = 1000,
        lineage: dict[str, str] | None = None,
        on_length=None,
    ):
        """`on_length(stream, length)` is called after each publish with the
        stream's length, for the metrics (ml_result_stream_length). It must not
        raise; a failure in it is ignored."""
        self._redis = redis
        self._stream = stream
        self._maxlen = maxlen
        self._lineage = dict(lineage or {})
        self._on_length = on_length

    async def publish(self, data: dict, text: str) -> None:
        result = {
            "document_id": str(data.get("document_id", "")),
            "text": text,
            "text_length": str(len(text)),
            **self._lineage,
        }
        stream = f"{self._stream}:results"

        # maxlen is enforced approximately: redis-py's xadd defaults to
        # approximate=True, so Redis trims in whole macro-nodes and the stream
        # can sit slightly above the cap.
        await self._redis.xadd(stream, result, maxlen=self._maxlen)

        # Once the stream is at the cap, every publish evicts the oldest
        # result — silently, from the producer's side. That is the only
        # warning an operator gets that the backend consumer is behind.
        if self._maxlen:
            length = await self._redis.xlen(stream)
            if self._on_length is not None:
                try:
                    self._on_length(stream, length)
                except Exception:
                    logger.debug("on_length callback failed", exc_info=True)
            if length >= self._maxlen:
                logger.warning(
                    "Results stream %s is at its %d-entry cap — the consumer is "
                    "behind and the oldest results are being dropped.",
                    stream,
                    self._maxlen,
                )
