# ADR-004: OCR multimodal via VLM

**Status:** ✅ Aceito  
**Data:** 2026-06-02  

## Contexto

O OCR precisa extrair texto de imagens de documentos (BOs, inquéritos). A abordagem tradicional seriam modelos dedicados de OCR (Tesseract, TrOCR).

## Decisão

Usar um **VLM (Vision Language Model)** do HuggingFace em vez de um OCR tradicional. O worker `ocr_worker.py` usa `AutoModelForImageTextToText` com um system prompt configurável no payload.

- **Modelo:** `Dharma-AI/Dharma-OCR-LITE` (configurável via `ocr_model_name`)
- **Pipeline:** Carregamento lazy → download da imagem (URL, base64 ou path) → chat template com mensagens system/user → geração de texto
- **Prompt:** System + User fixos em português (extração de texto e tabelas)

## Consequências

- **Positivas:** Único modelo lida com OCR + extração de tabelas +理解 contextual; system prompt flexível para diferentes tipos de documento; pipeline unificado.
- **Negativas:** Modelo mais pesado que um OCR dedicado (TrOCR); latência maior; requer GPU para tempo real.
- **Alternativas rejeitadas:** Tesseract (baixa qualidade em documentos complexos), TrOCR (apenas texto, sem tabelas), Docling (mais uma dependência).
