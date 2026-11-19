"""GLM-OCR, the alternative OCR model."""
from models.ocr import prompts
from models.ocr.vllm_chat_base import VllmChatOcrModel


class GlmOcrModel(VllmChatOcrModel):
    """`zai-org/GLM-OCR` on vLLM. Select it with `OCR_MODEL_NAME=zai-org/GLM-OCR`.

    It scored highest of 8 models on the Portuguese MDPBench samples (Overall
    85.1, Dharma 56.0), but it is not the default. See
    docs/adr-004-multimodal-ocr.md.

    It answers with plain text, so it keeps the base class's `_postprocess`.
    It runs on the GPU only, and it takes full-size images without resizing.
    """

    SYSTEM_PROMPT = prompts.SYSTEM_PROMPT
    USER_PROMPT = prompts.GLM_USER_PROMPT
