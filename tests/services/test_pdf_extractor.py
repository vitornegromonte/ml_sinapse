import io
from unittest.mock import patch

import fitz
import pytest
from PIL import Image

from services.pdf_extractor import PdfExtractor

FULL_PAGE = "[Página Escaneada Integralmente]"
BLOCK = "[Anexo Escaneado Intercalado]"
LONG_TEXT = "texto da pagina com mais de cinquenta caracteres para ser lido. " * 2


def _png(width: int = 300, height: int = 200) -> bytes:
    buf = io.BytesIO()
    Image.effect_noise((width, height), 90).convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def mixed_pdf(tmp_path):
    """Page 1 text only; 2 scanned; 3 text + large image; 4 scanned;
    5 text + tiny image (must be ignored)."""
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), LONG_TEXT)

    doc.new_page().insert_image(fitz.Rect(0, 0, 595, 842), stream=_png(600, 800))

    page = doc.new_page()
    page.insert_text((72, 72), LONG_TEXT)
    page.insert_image(fitz.Rect(72, 200, 372, 400), stream=_png())

    doc.new_page().insert_image(fitz.Rect(0, 0, 595, 842), stream=_png(600, 800))

    page = doc.new_page()
    page.insert_text((72, 72), LONG_TEXT)
    page.insert_image(fitz.Rect(72, 200, 122, 250), stream=_png(50, 50))

    path = tmp_path / "mixed.pdf"
    doc.save(path)
    return path


async def _collect(extractor, path, **kwargs):
    return [chunk async for chunk in extractor.extract_chunks(path, **kwargs)]


class TestExtractChunks:
    async def test_scanned_chunks_come_in_page_order_and_tiny_images_are_ignored(self, mixed_pdf):
        """Arrange: a PDF with 2 scanned pages, 1 large image block, and 1 tiny image.
        Act: extract its chunks.
        Assert: three image chunks in page order with their tags; the tiny
        image yields nothing.
        """
        chunks = await _collect(PdfExtractor(), mixed_pdf)

        scanned = [(kind, tag) for kind, _, tag in chunks if kind != "text"]
        assert scanned == [("image", FULL_PAGE), ("image", BLOCK), ("image", FULL_PAGE)]

    async def test_cached_chunks_are_not_rendered(self, mixed_pdf):
        """Arrange: scanned chunks 0 and 2 are marked as cached.
        Act: extract the chunks with rendering watched.
        Assert: those two come back as ("cached", "", tag), only chunk 1 is
        rendered, and the chunk count is the same as without a cache.
        """
        extractor = PdfExtractor()
        uncached = await _collect(extractor, mixed_pdf)

        with (
            patch.object(extractor, "_render_page", wraps=extractor._render_page) as page_render,
            patch.object(extractor, "_render_region", wraps=extractor._render_region) as region_render,
        ):
            chunks = await _collect(extractor, mixed_pdf, cached_images={0, 2})

        scanned = [(kind, content if kind == "cached" else "<img>", tag)
                   for kind, content, tag in chunks if kind != "text"]
        assert scanned == [("cached", "", FULL_PAGE), ("image", "<img>", BLOCK), ("cached", "", FULL_PAGE)]
        page_render.assert_not_called()
        region_render.assert_called_once()
        assert len(chunks) == len(uncached)

    async def test_numbering_ignores_tiny_images(self, mixed_pdf):
        """Arrange: chunk 3 does not exist (the tiny image is skipped).
        Act: extract with 3 and 99 marked as cached.
        Assert: nothing is returned as cached, so the numbering has no gap.
        """
        chunks = await _collect(PdfExtractor(), mixed_pdf, cached_images={3, 99})

        assert [kind for kind, _, _ in chunks if kind == "cached"] == []

    async def test_text_chunks_do_not_change_with_the_cache(self, mixed_pdf):
        """Arrange: the same PDF extracted with and without a cache.
        Act: compare the text chunks.
        Assert: they are identical.
        """
        extractor = PdfExtractor()
        plain = [c for c in await _collect(extractor, mixed_pdf) if c[0] == "text"]
        cached = [c for c in await _collect(extractor, mixed_pdf, cached_images={0, 1, 2}) if c[0] == "text"]

        assert plain == cached
        assert plain
