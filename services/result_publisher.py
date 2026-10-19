import logging

from redis.asyncio import Redis as AsyncRedis

logger = logging.getLogger(__name__)


class ResultPublisher:
    def __init__(self, redis: AsyncRedis, stream: str, maxlen: int = 1000):
        self._redis = redis
        self._stream = stream
        self._maxlen = maxlen

    async def publish(self, data: dict, text: str) -> None:
        result = {
            "document_id": str(data.get("document_id", "")),
            "text": text,
            "text_length": str(len(text)),
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
            if length >= self._maxlen:
                logger.warning(
                    "Results stream %s is at its %d-entry cap — the consumer is "
                    "behind and the oldest results are being dropped.",
                    stream,
                    self._maxlen,
                )
