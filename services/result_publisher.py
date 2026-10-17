from redis.asyncio import Redis as AsyncRedis


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
        await self._redis.xadd(
            f"{self._stream}:results",
            result,
            maxlen=self._maxlen,
        )
