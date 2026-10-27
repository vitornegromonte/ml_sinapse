#!/usr/bin/env python3
"""Valida pipeline PDF → OCR → LLM de ponta a ponta e mede a VRAM de cada etapa.

Uso:
    python scripts/test_ocr_llm.py documento.pdf

As utilizações vêm do config (as mesmas do compose), então dá para testar o
roteamento sem editar nada:
    OCR_GPU_MEMORY_UTILIZATION=0.25 LLM_GPU_MEMORY_UTILIZATION=0.25 \
        python scripts/test_ocr_llm.py documento.pdf

Cada modelo fica numa GPU (ocr na 0, llm na 1) e a medição é o delta de
`torch.cuda.mem_get_info` naquela placa — o contexto CUDA do processo aparece
separado do modelo, como em scripts/profile_models.py.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from models.ocr import create_model
from models.llm_model import LLMModel
from services.pdf_extractor import PdfExtractor
from services.skew_corrector import SkewCorrector
from services.image_loader import ImageLoader


def _vram(gpu: int) -> tuple[int, int]:
    """(usado, total) da placa, em MB."""
    import torch

    livre, total = torch.cuda.mem_get_info(gpu)
    return (total - livre) // 2**20, total // 2**20


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

    base_ocr, total_ocr = _vram(0)  # a primeira chamada cria o contexto: base ≈ contexto
    print(f"Carregando modelo OCR ({settings.ocr_model_name}, util {settings.ocr_gpu_memory_utilization}, max_len {settings.ocr_max_model_len or 'do modelo'})...")
    ocr = create_model(
        model_name=settings.ocr_model_name,
        cache_dir="/tmp/models",
        gpu_device=0,
        gpu_memory_utilization=settings.ocr_gpu_memory_utilization,
        max_model_len=settings.ocr_max_model_len,
    )
    ocr.load()
    print(f"  GPU 0: +{_vram(0)[0] - base_ocr} MB de modelo (contexto: {base_ocr} MB)")

    corrector = SkewCorrector()

    print(f"Extraindo texto do PDF: {pdf_path.name}...")
    full_text = await extract_text(pdf_path, ocr, corrector)
    ocr_pico = _vram(0)[0] - base_ocr
    print(f"  Total: {len(full_text)} chars extraídos")
    print(f"  GPU 0: +{ocr_pico} MB no pico da extração\n")

    base_llm, total_llm = _vram(1)
    llm_path = settings.llm_model_path or "google/gemma-4-E4B-it"
    print(
        f"Carregando modelo LLM ({llm_path}, util {settings.llm_gpu_memory_utilization}, "
        f"max_len {settings.llm_max_model_len or 'do modelo'})..."
    )
    llm = LLMModel(
        model_path=llm_path,
        device="cuda:1",
        gpu_memory_utilization=settings.llm_gpu_memory_utilization,
        max_model_len=settings.llm_max_model_len,
        sampling_params={"max_tokens": 1024, "temperature": 0.3},
    )
    llm.load()
    print(f"  GPU 1: +{_vram(1)[0] - base_llm} MB de modelo (contexto: {base_llm} MB)")

    print("Gerando resumo...\n")
    prompt = build_prompt(full_text)
    summary = llm.predict(prompt)[0]["text"]
    llm_pico = _vram(1)[0] - base_llm
    print(f"{'='*60}")
    print(f"RESUMO ({len(summary)} chars):")
    print(f"{'='*60}")
    print(summary)
    print(f"{'='*60}\n")

    llm.unload()
    ocr.unload()

    print("VRAM ocupada por worker (contexto + modelo + pico):")
    print(f"  GPU 0  ocr  {base_ocr + ocr_pico:>7} MB de {total_ocr} MB")
    print(f"  GPU 1  llm  {base_llm + llm_pico:>7} MB de {total_llm} MB")
    print("Pipeline concluído.")


if __name__ == "__main__":
    asyncio.run(main())
