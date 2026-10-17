import asyncio
from pathlib import Path
from typing import AsyncIterator, Literal, Tuple

import fitz
from PIL import Image


class PdfExtractor:
    def __init__(self, dpi: int = 150, text_threshold: int = 50):
        self._dpi = dpi
        self._text_threshold = text_threshold

    async def page_count(self, pdf_path: Path) -> int:
        """Cheap page-count check — callers use this to reject oversized PDFs
        before paying for full extraction (see OcrWorker._process_pdf)."""
        doc = await asyncio.to_thread(fitz.open, pdf_path)
        try:
            return doc.page_count
        finally:
            doc.close()

    async def extract_chunks(
        self,
        pdf_path: Path,
    ) -> AsyncIterator[Tuple[Literal["text", "image"], str | Image.Image, str]]:
        doc = await asyncio.to_thread(fitz.open, pdf_path)
        try:
            for page in doc:
                text_content = await asyncio.to_thread(page.get_text, "text")
                if len(text_content.strip()) < self._text_threshold:
                    if not page.rect.is_empty and page.rect.get_area() >= 10000:
                        img = await asyncio.to_thread(self._render_page, page)
                        yield ("image", img, "[Página Escaneada Integralmente]")
                    continue

                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if block["type"] == 0:
                        block_text = self._extract_text_from_block(block)
                        if block_text.strip():
                            yield ("text", block_text.strip(), "")
                    elif block["type"] == 1:
                        bbox = fitz.Rect(block["bbox"])
                        if bbox.is_empty or bbox.get_area() < 10000:
                            continue
                        img = await asyncio.to_thread(self._render_region, page, bbox)
                        if img.width >= 10 and img.height >= 10:
                            yield ("image", img, "[Anexo Escaneado Intercalado]")
        finally:
            doc.close()

    def _render_page(self, page: fitz.Page) -> Image.Image:
        pix = page.get_pixmap(clip=page.rect, dpi=self._dpi)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    def _render_region(self, page: fitz.Page, bbox: fitz.Rect) -> Image.Image:
        pix = page.get_pixmap(clip=bbox, dpi=self._dpi)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    @staticmethod
    def _extract_text_from_block(block: dict) -> str:
        lines = []
        for line in block["lines"]:
            spans = [span["text"] for span in line["spans"]]
            lines.append(" ".join(spans))
        return "\n".join(lines)
