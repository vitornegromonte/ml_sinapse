from models.ocr.vllm_chat_base import VllmChatOcrModel


class DharmaOcrModel(VllmChatOcrModel):
    """Dharma-OCR (Qwen2.5-VL) served via vLLM.

    `Dharma-AI/Dharma-OCR-LITE` is a Qwen2.5-VL fine-tune (see
    docs/adr-004-multimodal-ocr.md). It is the default production OCR model
    (config.py's ocr_model_name). It scored lowest of 8 models on the
    Portuguese MDPBench samples (Overall 56.0 vs. GLM-OCR's 85.1); it was
    chosen anyway, so measure it on real BOs. Qwen2.5-VL reads images at
    their native size, so config.py's ocr_render_dpi sets the image token cost.
    """

    SYSTEM_PROMPT = (
        "Você é um assistente de transcrição de documentos, "
        "sua função é extrair todo o texto de uma imagem"
    )
    USER_PROMPT = (
        "Extraia todo o texto e todas as tabelas deste documento. "
        "Transcreva todas as tabelas caso existam multiplas."
    )
