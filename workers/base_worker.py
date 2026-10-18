from abc import ABC, abstractmethod
import asyncio
import json
import logging
import os
import signal
import socket
import time

from redis.asyncio import Redis as AsyncRedis
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from config import settings
from services.database import create_db_engine, create_session_factory


class BaseWorker(ABC):
    def __init__(self, stream: str, cfg=settings):
        self.stream = stream
        self.cfg = cfg

        hostname = socket.gethostname()
        pid = os.getpid()
        self.consumer_id = self.cfg.consumer_id or f"{hostname}-{pid}"
        self.group = self.cfg.consumer_group

        self.redis: AsyncRedis | None = None
        self.db_engine: AsyncEngine | None = None
        self.db_session_factory: async_sessionmaker[AsyncSession] | None = None
        self._running = True

        self.logger = logging.getLogger(
            f"{self.__class__.__name__}[{self.stream}]"
        )

    async def _connect_redis(self) -> AsyncRedis:
        # socket_timeout=None is load-bearing: redis-py's client-side read
        # timeout defaults to 5s, but XREADGROUP here blocks for up to
        # poll_timeout_ms (default 50s) server-side. With the 5s default, the
        # client raises TimeoutError on every idle poll — found live, against
        # a real Redis instance with an empty stream (this class was only ever
        # exercised against mocks or a stream with data already waiting, so
        # the mismatch never surfaced until now). retry_on_timeout=True does
        # NOT cover this — deprecated as a no-op since redis-py 6.0.
        return await AsyncRedis.from_url(
            self.cfg.redis_url,
            decode_responses=True,
            socket_keepalive=True,
            socket_connect_timeout=10,
            socket_timeout=None,
        )

    async def _ensure_group(self) -> None:
        try:
            await self.redis.xgroup_create(
                self.stream, self.group, id="0", mkstream=True
            )
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def _get_retries(self, message_id: str) -> int:
        key = f"{self.stream}:retries"
        val = await self.redis.hget(key, message_id)
        return int(val) if val else 0

    async def _increment_retries(self, message_id: str) -> int:
        key = f"{self.stream}:retries"
        return await self.redis.hincrby(key, message_id, 1)

    async def _cleanup_retries(self, message_id: str) -> None:
        key = f"{self.stream}:retries"
        await self.redis.hdel(key, message_id)

    def _lease_key(self, message_id: str) -> str:
        return f"{self.stream}:lease:{message_id}"

    async def _heartbeat(self, lease_key: str) -> None:
        """Keep a short-TTL lease alive while the message is in flight.

        Without it, XAUTOCLAIM can only distinguish "idle a while" from "still
        running" by time, so any task slower than claim_idle_ms gets stolen and
        processed twice (see _reclaim_stale)."""
        while True:
            await asyncio.sleep(self.cfg.lease_heartbeat_seconds)
            try:
                await self.redis.set(
                    lease_key, self.consumer_id, ex=self.cfg.lease_ttl_seconds
                )
            except Exception as exc:
                self.logger.warning(
                    "Lease heartbeat failed for %s: %s", lease_key, exc
                )

    async def _lease_alive(self, message_id: str) -> bool:
        return bool(await self.redis.exists(self._lease_key(message_id)))

    async def _dead_letter(
        self, message_id: str, data: dict, error: str
    ) -> None:
        dlq_stream = f"{self.stream}{self.cfg.dead_letter_suffix}"
        await self.redis.xadd(
            dlq_stream,
            {
                "original_id": message_id,
                "payload": json.dumps(data),
                "error": error,
                "timestamp": str(time.time()),
            },
            maxlen=self.cfg.dead_letter_maxlen,
            approximate=True,
        )
        await self.redis.expire(dlq_stream, self.cfg.dead_letter_ttl_seconds)
        self.logger.warning(
            "Moved %s to DLQ %s: %s", message_id, dlq_stream, error
        )

    @abstractmethod
    async def process(self, data: dict) -> None:
        ...

    async def _reclaim_stale(self) -> None:
        """Reclaim pending messages idle longer than claim_idle_ms and reprocess them.

        Covers two cases with the same mechanism: a message left un-acked by
        a not-yet-exhausted retry (see `_process_message`), and a message
        abandoned by a consumer that crashed mid-processing — XAUTOCLAIM scans
        the whole group's pending list, not just this consumer's own entries.
        """
        cursor = "0-0"
        try:
            while True:
                cursor, claimed, _ = await self.redis.xautoclaim(
                    self.stream,
                    self.group,
                    self.consumer_id,
                    min_idle_time=self.cfg.claim_idle_ms,
                    start_id=cursor,
                    count=self.cfg.batch_size,
                )
                for message_id, raw in claimed:
                    if await self._lease_alive(message_id):
                        self.logger.info(
                            "Not reclaiming %s — another consumer still holds "
                            "its lease",
                            message_id,
                        )
                        continue
                    await self._process_message(message_id, raw)
                if cursor == "0-0":
                    break
        except Exception as exc:
            self.logger.error("Error reclaiming stale pending messages: %s", exc)

    async def _process_message(
        self, message_id: str, data: dict
    ) -> None:
        lease_key = self._lease_key(message_id)
        # Set before processing so a concurrent reclaim sees the lease.
        await self.redis.set(
            lease_key, self.consumer_id, ex=self.cfg.lease_ttl_seconds
        )
        heartbeat = asyncio.create_task(self._heartbeat(lease_key))
        try:
            await self.process(data)
            await self.redis.xack(self.stream, self.group, message_id)
            await self._cleanup_retries(message_id)
        except Exception as exc:
            self.logger.error(
                "Error processing %s: %s", message_id, exc
            )
            retries = await self._increment_retries(message_id)
            if retries >= self.cfg.max_retries:
                await self._dead_letter(message_id, data, str(exc))
                await self.redis.xack(self.stream, self.group, message_id)
                await self._cleanup_retries(message_id)
            else:
                self.logger.info(
                    "Leaving %s pending for retry %d/%d — will be reclaimed by "
                    "XAUTOCLAIM once idle %dms (not acking it now)",
                    message_id,
                    retries,
                    self.cfg.max_retries,
                    self.cfg.claim_idle_ms,
                )
                # Deliberately NOT acking: an acked message can never be
                # redelivered by Redis Streams. Leaving it pending is what
                # lets _reclaim_stale() pick it back up for another attempt.
        finally:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass
            # Released either way: on success the message is acked; on failure
            # the retry must be reclaimable again after claim_idle_ms.
            await self.redis.delete(lease_key)

    async def start(self) -> None:
        self.redis = await self._connect_redis()
        await self._ensure_group()

        self.db_engine = create_db_engine(self.cfg)
        self.db_session_factory = create_session_factory(self.db_engine)

        self.logger.info(
            "Listening on stream=%s group=%s consumer=%s",
            self.stream,
            self.group,
            self.consumer_id,
        )

        while self._running:
            try:
                # Pick up messages left pending by a not-yet-exhausted retry
                # or an abandoned consumer before reading new ones.
                await self._reclaim_stale()

                results = await self.redis.xreadgroup(
                    self.group,
                    self.consumer_id,
                    {self.stream: ">"},
                    count=self.cfg.batch_size,
                    block=self.cfg.poll_timeout_ms,
                )
                if results:
                    for _, messages in results:
                        for message_id, raw in messages:
                            await self._process_message(message_id, raw)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.logger.error("Redis connection error: %s", exc)
                await asyncio.sleep(1)

        await self.db_engine.dispose()

        await self.redis.close()
        self.logger.info("Worker stopped")

    def _handle_signal(self) -> None:
        self.logger.info("Shutdown signal received")
        self._running = False

    def run(self) -> None:
        logging.basicConfig(
            level=getattr(logging, self.cfg.log_level),
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        )
        self.logger.info(
            "Starting worker process (pid=%s stream=%s)",
            os.getpid(),
            self.stream,
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_signal)

        try:
            loop.run_until_complete(self.start())
        except KeyboardInterrupt:
            pass
        finally:
            loop.close()
            self.logger.info("Worker process finished")
