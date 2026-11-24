# ADR-004: OCR multimodal via VLM

**Status:** ✅ Aceito (atualizado em 2026-10-28)<br>
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

## Atualização (2026-10-28)

A decisão de usar um VLM continua; o backend de execução mudou. O
`ocr_worker` usa `models/ocr/registry.py::create_model`, que escolhe
Dharma, GLM ou MLflow. Dharma e GLM rodam via **vLLM** (`LLM.chat`), não via
`AutoModelForImageTextToText`. O default do compose é `zai-org/GLM-OCR`;
`Dharma-AI/Dharma-OCR-LITE` continua suportado. O caminho lazy → imagem →
chat template → geração de texto permanece o mesmo.

## Atualização (2026-11-19)

O default voltou a ser `Dharma-AI/Dharma-OCR-LITE` (`config.py` e
`docker-compose.yml`). No benchmark MDPBench em português, o Dharma teve 56,0
Overall contra 85,1 do GLM-OCR; a troca foi uma decisão explícita do projeto e
deve ser validada em BOs reais. `zai-org/GLM-OCR` continua registrado: basta
definir `OCR_MODEL_NAME`.

O DPI de renderização das páginas escaneadas agora é configurável
(`ocr_render_dpi` / `OCR_RENDER_DPI`, padrão 200, limite 72–400). Antes era 150
fixo em `PdfExtractor`. O Qwen2.5-VL lê a imagem no tamanho original, então o
número de tokens de imagem cresce com o quadrado do DPI: uma página A4 tem
cerca de 2,8k tokens a 150 dpi, 4,9k a 200 dpi e 11k a 300 dpi. Se subir o DPI,
confira `OCR_MAX_MODEL_LEN` e `OCR_GPU_MEMORY_UTILIZATION`.

## Atualização (2026-11-19, saída JSON e limites de recurso)

O teste real com o modelo na GPU mostrou três ajustes:

- **Saída em JSON.** O Dharma-OCR-LITE responde cada página com
  `{"header", "margin", "footer", "text"}` (campos ausentes vêm `null`), não com
  texto puro. `DharmaOcrModel` agora devolve só o campo `text`, como texto
  simples. `header`, `margin` e `footer` são descartados como ruído de página.
  Se o JSON vier cortado por `max_tokens`, o wrapper devolve o trecho de `text`
  que foi escrito e registra um aviso. Saída que não é JSON passa sem mudança.
  O GLM-OCR não é afetado. O conteúdo da página nunca vai para o log. O modelo
  pode repetir cabeçalho ou rodapé dentro de `text`; isso não é tratado aqui.
- **Páginas por passada.** Esta seção chegou a definir `ocr_page_batch_size`
  com padrão 1. A medição do ADR-012 mostrou que o batching é o maior ganho de
  velocidade, e o padrão passou a 4 (`OCR_PAGE_BATCH_SIZE=1` volta a uma página
  por chamada). O timeout do ADR-011 vale por chamada ao modelo.
- **Teto de 15% da GPU.** `ocr_gpu_memory_utilization` tem padrão e máximo de
  0,15; um valor maior falha na inicialização. A GPU é dividida com os workers
  de embedding e linkage. Os pesos ocupam 4,6 GiB e o cache KV custa cerca de
  36 KiB por token. Medido na GPU (95 GiB): o motor ocupa +13 GB, com 6 GiB de
  cache KV (176 mil tokens).
- **Contexto de 24 576 e sem vídeo.** Com o contexto nativo de 128 000 o vLLM
  perfila a memória com uma entrada de vídeo de tamanho máximo, reserva cerca
  de 21 GiB e não inicia em 15% (cache KV disponível: −6,7 GiB).
  `ocr_max_model_len` passou a ter padrão 24 576: uma página tem no máximo
  16 384 tokens de imagem (`max_pixels` do processador) mais `ocr_max_tokens` de
  saída. `DharmaOcrModel` também desliga vídeo (`limit_mm_per_prompt`
  `{"image": 1, "video": 0}`). O GLM-OCR continua só com `{"image": 1}`.
