from models.ocr.vllm_chat_base import VllmChatOcrModel


class GlmOcrModel(VllmChatOcrModel):
    """GLM-OCR served via vLLM.

    Migrated from a transformers .generate() implementation.
    Benchmarked as the strongest of 8 OCR models
    tested against Portuguese MDPBench samples (Overall 85.1 vs. Dharma's
    56.0), and is now the default production OCR model
    (config.py's ocr_model_name).

    No CPU fallback and no image-resize step here, unlike the transformers
    version this replaced — vLLM is GPU-only, and the benchmark ran this
    exact model through vLLM's LLM.chat() with full-size images with no
    resizing needed.
    """

    SYSTEM_PROMPT = (
        "Você é um assistente de transcrição de documentos, "
        "sua função é extrair todo o texto de uma imagem"
    )
    USER_PROMPT = (
        "Transcreva todo o texto presente nesta imagem com extrema "
        "precisão, no idioma português do Brasil."
    )
