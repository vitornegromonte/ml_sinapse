import pytest

from services.ocr_checkpoint import OcrCheckpoint, checkpoint_key


class _FakeRedis:
    def __init__(self):
        self.hashes: dict[str, dict[str, str]] = {}
        self.ttls: dict[str, int] = {}
        self.calls: list[str] = []

    async def hgetall(self, key):
        self.calls.append("hgetall")
        return dict(self.hashes.get(key, {}))

    async def hset(self, key, mapping):
        self.calls.append("hset")
        self.hashes.setdefault(key, {}).update(mapping)

    async def expire(self, key, ttl):
        self.calls.append("expire")
        self.ttls[key] = ttl

    async def delete(self, key):
        self.calls.append("delete")
        self.hashes.pop(key, None)


class _BrokenRedis:
    async def hgetall(self, key):
        raise ConnectionError("boom")

    async def hset(self, key, mapping):
        raise ConnectionError("boom")

    async def expire(self, key, ttl):
        raise ConnectionError("boom")

    async def delete(self, key):
        raise ConnectionError("boom")


class TestCheckpointKey:
    def test_same_inputs_give_the_same_key(self):
        """Arrange: the same stream, document, PDF bytes, and settings twice.
        Act: build the key twice.
        Assert: the keys are equal, so a retry finds its own pages.
        """
        first = checkpoint_key("ocr:tasks", "doc-1", b"%PDF-a", "model|200|4096")
        second = checkpoint_key("ocr:tasks", "doc-1", b"%PDF-a", "model|200|4096")

        assert first == second
        assert first.startswith("ocr:tasks:partial:doc-1:")

    @pytest.mark.parametrize("stream, document_id, pdf, tag", [
        ("ocr:tasks", "doc-2", b"%PDF-a", "model|200|4096"),
        ("ocr:tasks", "doc-1", b"%PDF-b", "model|200|4096"),
        ("ocr:tasks", "doc-1", b"%PDF-a", "model|300|4096"),
        ("ocr:tasks", "doc-1", b"%PDF-a", "other|200|4096"),
        ("ocr:tasks", "doc-1", b"%PDF-a", "model|200|2048"),
    ])
    def test_any_change_gives_a_new_key(self, stream, document_id, pdf, tag):
        """Arrange: a base key, and a variant that changes the document, the
        PDF bytes, the DPI, the model, or the token cap.
        Act: build the variant's key.
        Assert: it differs, so stale text is never reused.
        """
        base = checkpoint_key("ocr:tasks", "doc-1", b"%PDF-a", "model|200|4096")

        variant = checkpoint_key(stream, document_id, pdf, tag)

        assert variant != base

    def test_missing_document_id_still_gives_a_usable_key(self):
        """Arrange: an empty document_id.
        Act: build the key.
        Assert: it has a readable placeholder instead of an empty segment.
        """
        key = checkpoint_key("ocr:tasks", "", b"%PDF", "t")

        assert ":partial:no-id:" in key


class TestOcrCheckpoint:
    async def test_saved_pages_load_back_with_integer_keys(self):
        """Arrange: a checkpoint on a fake Redis.
        Act: save two pages, then load.
        Assert: the pages come back keyed by integer, and the TTL is set.
        """
        redis = _FakeRedis()
        checkpoint = OcrCheckpoint(redis, "k", ttl_seconds=600)

        await checkpoint.save({0: "pagina zero", 3: "pagina tres"})
        loaded = await checkpoint.load()

        assert loaded == {0: "pagina zero", 3: "pagina tres"}
        assert redis.ttls["k"] == 600

    async def test_later_saves_add_to_earlier_ones(self):
        """Arrange: a checkpoint with one saved page.
        Act: save another page.
        Assert: both pages are kept.
        """
        checkpoint = OcrCheckpoint(_FakeRedis(), "k", ttl_seconds=600)
        await checkpoint.save({0: "a"})

        await checkpoint.save({1: "b"})

        assert await checkpoint.load() == {0: "a", 1: "b"}

    async def test_saving_nothing_does_not_touch_redis(self):
        """Arrange: a checkpoint on a fake Redis.
        Act: save an empty mapping.
        Assert: no Redis call is made.
        """
        redis = _FakeRedis()

        await OcrCheckpoint(redis, "k", 600).save({})

        assert redis.calls == []

    async def test_clear_removes_the_pages(self):
        """Arrange: a checkpoint with a saved page.
        Act: clear it.
        Assert: loading returns nothing.
        """
        checkpoint = OcrCheckpoint(_FakeRedis(), "k", 600)
        await checkpoint.save({0: "a"})

        await checkpoint.clear()

        assert await checkpoint.load() == {}

    async def test_redis_faults_never_raise(self, caplog):
        """Arrange: a Redis that fails every call, and a recognizable page text.
        Act: load, save, and clear.
        Assert: nothing raises, load returns an empty dict, and the page text
        is not in the logs.
        """
        checkpoint = OcrCheckpoint(_BrokenRedis(), "k", 600)

        with caplog.at_level("WARNING"):
            loaded = await checkpoint.load()
            await checkpoint.save({0: "texto-sensivel-do-bo"})
            await checkpoint.clear()

        assert loaded == {}
        assert "texto-sensivel-do-bo" not in caplog.text
        assert caplog.text.count("Could not") == 3
