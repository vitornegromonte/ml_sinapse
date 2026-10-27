#!/usr/bin/env python3
"""Valida pipeline PDF → OCR → LLM com uma única execução na GPU."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ocr import create_model
from models.llm_model import LLMModel
from services.pdf_extractor import PdfExtractor
from services.skew_corrector import SkewCorrector
from services.image_loader import ImageLoader


PROMPT_TEMPLATE = (
    "Você é um assistente especializado em analisar documentos policiais.\n"
    "Abaixo está o texto extraído de um documento:\n\n"
    "{text}\n\n"
    "Produza um resumo conciso destacando:\n"
    "1. Fatos principais\n"
    "2. Envolvidos (se houver)\n"
    "3. Datas e locais mencionados\n"
    "4. Provas documentais citadas\n\n"
    "Resumo:"
)


async def extract_text(pdf_path: Path, ocr, corrector) -> str:
    """Extrai texto do PDF: texto direto + OCR em páginas/regiões escaneadas."""
    chunks = []
    extractor = PdfExtractor(dpi=150)
    async for kind, content, _ in extractor.extract_chunks(pdf_path):
        if kind == "text":
            chunks.append(content)
        else:
            img = corrector.correct(content)
            text = ocr.predict(img, 2048)
            chunks.append(text)
            print(f"  [OCR] imagem processada ({len(text)} chars)")
    return "\n".join(chunks)


def build_prompt(text: str) -> str:
    return PROMPT_TEMPLATE.format(text=text)


async def main():
    if len(sys.argv) < 2:
        print("Uso: python scripts/test_ocr_llm.py <caminho-do-pdf>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"Arquivo não encontrado: {pdf_path}")
        sys.exit(1)

    print("Carregando modelo OCR (GLM-OCR)...")
    ocr = create_model(
        model_name="zai-org/GLM-OCR",  # default de produção — ver docs/adr-004-multimodal-ocr.md
        cache_dir="/tmp/models",
        gpu_device=0,
    )
    ocr.load()

    corrector = SkewCorrector()

    print(f"Extraindo texto do PDF: {pdf_path.name}...")
    full_text = await extract_text(pdf_path, ocr, corrector)
    print(f"  Total: {len(full_text)} chars extraídos\n")

    print("Carregando modelo LLM (Gemma 4 via vLLM)...")
    llm = LLMModel(
        model_path="google/gemma-4-E4B-it",
        device="cuda:1",
        gpu_memory_utilization=0.9,
        max_model_len=32768,
        sampling_params={"max_tokens": 1024, "temperature": 0.3},
    )
    llm.load()

    print("Gerando resumo...\n")
    prompt = build_prompt(full_text)
    summary = llm.predict(prompt)[0]["text"]
    print(f"{'='*60}")
    print(f"RESUMO ({len(summary)} chars):")
    print(f"{'='*60}")
    print(summary)
    print(f"{'='*60}\n")

    llm.unload()
    ocr.unload()
    print("Pipeline concluído.")


if __name__ == "__main__":
    asyncio.run(main())
