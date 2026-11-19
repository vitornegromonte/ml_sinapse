"""Shared helpers for the OcrWorker tests: a fake Redis and fake PDFs.

They are plain functions, not fixtures, so a test can pass the arguments it
needs. Fixtures are in conftest.py.
"""
from contextlib import contextmanager
from unittest.mock import patch


TAG = "[Anexo Escaneado Intercalado]"
PDF_TASK = {"document_id": "doc-r", "pdf_bucket": "bronze", "pdf_key": "k.pdf"}


class FakeRedis:
    """In-memory Redis: just the calls OcrWorker, ResultPublisher and the
    checkpoint make. `fail_xadd` makes the result publish fail."""

    def __init__(self, fail_xadd: bool = False):
        self.hashes: dict[str, dict[str, str]] = {}
        self.deleted: list[str] = []
        self._fail_xadd = fail_xadd

    async def xadd(self, *args, **kwargs):
        if self._fail_xadd:
            raise ConnectionError("redis down")
        return "1-0"

    async def xlen(self, *args, **kwargs):
        return 0

    async def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    async def hset(self, key, mapping):
        self.hashes.setdefault(key, {}).update(mapping)

    async def expire(self, key, ttl):
        pass

    async def delete(self, key):
        self.hashes.pop(key, None)
        self.deleted.append(key)


@contextmanager
def scanned_pdf(worker, pages, seen_cached=None, on_yield=None):
    """Make a worker see a 'PDF' of one text chunk and `pages` scanned pages.
    The fake extractor honors cached_images like the real PdfExtractor."""

    async def fake_chunks(pdf_path, cached_images=()):
        if seen_cached is not None:
            seen_cached.append(set(cached_images))
        yield ("text", "inicio", "")
        for number in range(pages):
            if on_yield is not None:
                on_yield(number)
            if number in cached_images:
                yield ("cached", "", TAG)
            else:
                yield ("image", f"img{number}", TAG)

    with (
        patch("workers.ocr_worker.StorageService") as storage_cls,
        patch.object(worker._pdf_extractor, "page_count", return_value=pages),
        patch.object(worker._pdf_extractor, "extract_chunks", side_effect=fake_chunks),
        patch("workers.ocr_worker.SkewCorrector.correct", side_effect=lambda img: img),
    ):
        storage_cls.return_value.download.return_value = b"%PDF-same-bytes"
        worker._storage = storage_cls.return_value
        yield


def echo_model(model, images_seen, fail_on_call=None):
    calls = {"n": 0}

    def predict_batch(images, max_tokens, prompt=None):
        calls["n"] += 1
        if fail_on_call == calls["n"]:
            raise RuntimeError("gpu fault")
        images_seen.append(list(images))
        return [f"OCR:{image}" for image in images]

    model.predict_batch.side_effect = predict_batch


def run_pdf(worker, chunks):
    """Four patches that make `worker` see a PDF made of exactly `chunks`.

    Enter them all with `with`. The fake extractor ignores `cached_images`.
    """

    async def fake_chunks(pdf_path, cached_images=()):
        for chunk in chunks:
            yield chunk

    return (
        patch("workers.ocr_worker.StorageService"),
        patch.object(worker._pdf_extractor, "page_count", return_value=len(chunks)),
        patch.object(worker._pdf_extractor, "extract_chunks", side_effect=fake_chunks),
        patch("workers.ocr_worker.SkewCorrector.correct", side_effect=lambda img: img),
    )
