from models.ocr.vllm_chat_base import VllmChatOcrModel


class DharmaOcrModel(VllmChatOcrModel):
    """Dharma-OCR (Qwen2.5-VL) served via vLLM.

    Same model weights as before (`Dharma-AI/Dharma-OCR-LITE`, a Qwen2.5-VL
    fine-tune, see docs/adr-004-multimodal-ocr.md). No longer the
    default production OCR model — benchmarked
    lowest of 8 models tested against Portuguese MDPBench samples (Overall
    56.0 vs. GLM-OCR's 85.1). Kept registered/selectable via ocr_model_name.
    """

    SYSTEM_PROMPT = (
        "Você é um assistente de transcrição de documentos, "
        "sua função é extrair todo o texto de uma imagem"
    )
    USER_PROMPT = (
        "Extraia todo o texto e todas as tabelas deste documento. "
        "Transcreva todas as tabelas caso existam multiplas."
    )
