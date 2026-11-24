"""Split a PDF into text chunks and scanned-image chunks, in reading order."""
import asyncio
import itertools
from pathlib import Path
from collections.abc import AsyncIterator, Container, Iterator
from typing import Literal

import fitz
from PIL import Image

# PDF geometry is in points: 72 points = 1 inch.
_POINTS_PER_INCH = 72

# A page or image smaller than this (in square points, about 3.5 x 3.5 cm) is
# not worth OCR. It is usually a logo, a stamp or a speck.
_MIN_REGION_AREA_PT2 = 10_000

# A rendered image thinner than this (in pixels) cannot hold readable text.
_MIN_RENDERED_SIDE_PX = 10

# Tags that mark scanned text in the output, so a reader of the OCR text can
# tell it from the text layer.
FULL_PAGE_TAG = "[Página Escaneada Integralmente]"
EMBEDDED_IMAGE_TAG = "[Anexo Escaneado Intercalado]"

# "text": the text layer of a page block.
# "image": a scanned region, rendered and ready for OCR.
# "cached": a scanned region that is not rendered, because the caller already
#   has its text (see OcrCheckpoint). Its content is "".
ChunkKind = Literal["text", "image", "cached"]
Chunk = tuple[ChunkKind, str | Image.Image, str]  # (kind, content, tag)


class PdfExtractor:
    """Reads a PDF with PyMuPDF. Blocking calls run in threads.

    Args:
        dpi: resolution for rendering scanned regions. Image tokens grow with
            its square, so it sets the OCR cost.
        text_threshold: a page whose text layer has fewer characters than this
            counts as a scan.
    """

    def __init__(self, dpi: int = 150, text_threshold: int = 50):
        self._dpi = dpi
        self._text_threshold = text_threshold

    async def page_count(self, pdf_path: Path) -> int:
        """Return the number of pages without extracting anything.

        Callers use it to reject an oversized PDF before the full extraction
        (see OcrWorker._process_pdf).
        """
        doc = await asyncio.to_thread(fitz.open, pdf_path)
        try:
            return doc.page_count
        finally:
            doc.close()

    async def extract_chunks(
        self,
        pdf_path: Path,
        cached_images: Container[int] = frozenset(),
    ) -> AsyncIterator[Chunk]:
        """Yield the PDF as chunks, in page order.

        A page with a text layer gives one "text" chunk per text block, and one
        scanned chunk per large embedded image. A page without a text layer
        gives one scanned chunk for the whole page.

        Scanned chunks are numbered 0, 1, 2... in the order they appear. The
        numbering depends only on the PDF and on this extractor's settings, so
        it is the same on every run. A chunk whose number is in `cached_images`
        is yielded as ("cached", "", tag) and is not rendered: the caller
        already has its text.
        """
        doc = await asyncio.to_thread(fitz.open, pdf_path)
        numbers = itertools.count()
        try:
            for page in doc:
                text_layer = await asyncio.to_thread(page.get_text, "text")
                if len(text_layer.strip()) >= self._text_threshold:
                    async for chunk in self._chunks_of_text_page(
                        page, numbers, cached_images
                    ):
                        yield chunk
                elif self._is_large_enough(page.rect):
                    yield await self._scanned_chunk(
                        next(numbers), cached_images, FULL_PAGE_TAG, self._render_page, page
                    )
        finally:
            doc.close()

    # private

    async def _chunks_of_text_page(
        self, page: fitz.Page, numbers: Iterator[int], cached_images: Container[int]
    ) -> AsyncIterator[Chunk]:
        """Yield the text blocks of a page, and its large embedded images."""
        for block in page.get_text("dict")["blocks"]:
            if block["type"] == 0:  # text block
                block_text = self._block_text(block).strip()
                if block_text:
                    yield ("text", block_text, "")
            elif block["type"] == 1:  # image block
                bbox = fitz.Rect(block["bbox"])
                if not self._is_large_enough(bbox) or not self._is_thick_enough(bbox):
                    continue
                yield await self._scanned_chunk(
                    next(numbers), cached_images, EMBEDDED_IMAGE_TAG,
                    self._render_region, page, bbox,
                )

    async def _scanned_chunk(
        self, number: int, cached_images: Container[int], tag: str, render, *render_args
    ) -> Chunk:
        """Render scanned chunk `number`, or return a placeholder if it is cached."""
        if number in cached_images:
            return ("cached", "", tag)
        image = await asyncio.to_thread(render, *render_args)
        return ("image", image, tag)

    @staticmethod
    def _is_large_enough(rect: fitz.Rect) -> bool:
        return not rect.is_empty and rect.get_area() >= _MIN_REGION_AREA_PT2

    def _is_thick_enough(self, bbox: fitz.Rect) -> bool:
        """Check the rendered size from the geometry, before rendering.

        Doing it before keeps the chunk numbers independent of the cache: a
        cached chunk is never rendered, but it is still counted.
        """
        rendered_side_px = min(bbox.width, bbox.height) * self._dpi / _POINTS_PER_INCH
        return rendered_side_px >= _MIN_RENDERED_SIDE_PX

    def _render_page(self, page: fitz.Page) -> Image.Image:
        return self._render(page, page.rect)

    def _render_region(self, page: fitz.Page, bbox: fitz.Rect) -> Image.Image:
        return self._render(page, bbox)

    def _render(self, page: fitz.Page, clip: fitz.Rect) -> Image.Image:
        pixmap = page.get_pixmap(clip=clip, dpi=self._dpi)
        return Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)

    @staticmethod
    def _block_text(block: dict) -> str:
        """The text of a text block: spans joined by spaces, lines by newlines."""
        return "\n".join(
            " ".join(span["text"] for span in line["spans"]) for line in block["lines"]
        )
