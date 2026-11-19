import hashlib
import logging

from redis.asyncio import Redis as AsyncRedis

logger = logging.getLogger(__name__)


def checkpoint_key(stream: str, document_id: str, pdf_bytes: bytes, settings_tag: str) -> str:
    """Redis key for the finished pages of one OCR run.

    The key includes a hash of the PDF bytes and of the settings that change
    the OCR text (model, DPI, token cap). A retry with a changed PDF or changed
    settings gets a new key, so stale text is never reused.
    """
    digest = hashlib.sha256(pdf_bytes)
    digest.update(settings_tag.encode("utf-8"))
    return f"{stream}:partial:{document_id or 'no-id'}:{digest.hexdigest()[:24]}"


class OcrCheckpoint:
    """Finished scanned pages of one PDF, kept in a Redis hash with a TTL.

    The hash maps the scanned-chunk number (see PdfExtractor.extract_chunks) to
    its OCR text. Every method is best effort: a Redis fault is logged and
    skipped, because a missing checkpoint only costs a repeat of the OCR work,
    while a failed checkpoint must never fail the task. Page text is never
    logged.
    """

    def __init__(self, redis: AsyncRedis, key: str, ttl_seconds: int):
        self._redis = redis
        self._key = key
        self._ttl = ttl_seconds

    async def load(self) -> dict[int, str]:
        try:
            raw = await self._redis.hgetall(self._key)
            return {int(number): text for number, text in raw.items()}
        except Exception as exc:
            logger.warning("Could not read OCR checkpoint %s: %s", self._key, exc)
            return {}

    async def save(self, pages: dict[int, str]) -> None:
        if not pages:
            return
        try:
            await self._redis.hset(
                self._key, mapping={str(number): text for number, text in pages.items()}
            )
            await self._redis.expire(self._key, self._ttl)
        except Exception as exc:
            logger.warning("Could not write OCR checkpoint %s: %s", self._key, exc)

    async def clear(self) -> None:
        try:
            await self._redis.delete(self._key)
        except Exception as exc:
            logger.warning("Could not delete OCR checkpoint %s: %s", self._key, exc)
