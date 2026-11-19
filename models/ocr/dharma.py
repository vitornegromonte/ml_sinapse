import json
import logging
import re

from models.ocr.vllm_chat_base import VllmChatOcrModel

logger = logging.getLogger(__name__)

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
_TEXT_FIELD = re.compile(r'"text"\s*:\s*"')


class DharmaOcrModel(VllmChatOcrModel):
    """Dharma-OCR (Qwen2.5-VL) served via vLLM.

    `Dharma-AI/Dharma-OCR-LITE` is a Qwen2.5-VL fine-tune (see
    docs/adr-004-multimodal-ocr.md). It is the default production OCR model
    (config.py's ocr_model_name). It scored lowest of 8 models on the
    Portuguese MDPBench samples (Overall 56.0 vs. GLM-OCR's 85.1); it was
    chosen anyway, so measure it on real BOs. Qwen2.5-VL reads images at
    their native size, so config.py's ocr_render_dpi sets the image token cost.

    The model answers each page with a JSON object:
    {"header": ..., "margin": ..., "footer": ..., "text": "..."}, with null for
    regions the page does not have. predict() and predict_batch() return only
    the `text` field as plain text. header, margin and footer are dropped as
    page noise.
    """

    # Qwen2.5-VL also accepts video. Turning it off removes vLLM's profiling run
    # for a maximum-size video, which needed ~20 GiB at the native 128k context
    # and made the 15% GPU cap (config.py) impossible to meet.
    MM_LIMITS = {"image": 1, "video": 0}

    # The model's own generation_config.json sets repetition_penalty 1.05. Our
    # explicit SamplingParams ignores that file, and without it some pages
    # looped until max_tokens (2 of 5 scanned pages on a test PDF).
    REPETITION_PENALTY = 1.05

    SYSTEM_PROMPT = (
        "Você é um assistente de transcrição de documentos, "
        "sua função é extrair todo o texto de uma imagem"
    )
    USER_PROMPT = (
        "Extraia todo o texto e todas as tabelas deste documento. "
        "Transcreva todas as tabelas caso existam multiplas."
    )

    def _postprocess(self, text: str) -> str:
        """Return the page's body text from the model's JSON answer.

        Falls back, in order: a fenced ```json block is unwrapped; valid JSON
        gives its `text` field; broken JSON (for example cut off by max_tokens)
        gives whatever of the `text` string was written; anything else is plain
        text and is returned unchanged. Page content is never logged.
        """
        raw = text.strip()
        fenced = _FENCE.match(raw)
        if fenced:
            raw = fenced.group(1)

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return self._salvage_text(raw)

        if isinstance(parsed, dict) and "text" in parsed:
            body = parsed["text"]
            return body.strip() if isinstance(body, str) else ""
        logger.warning("Dharma output is JSON without a 'text' field; keeping it as is")
        return raw

    @staticmethod
    def _salvage_text(raw: str) -> str:
        """Read the `text` string out of JSON that does not parse as a whole.

        Reads up to the closing quote, or to the end when the output was cut
        off. Without a `"text":` field the output is plain text: return it.
        """
        match = _TEXT_FIELD.search(raw)
        if match is None:
            return raw

        body = raw[match.end():]
        closed, escaped = False, False
        for index, char in enumerate(body):
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                body, closed = body[:index], True
                break

        if not closed:
            logger.warning("Dharma output ended inside the 'text' field; page text is partial")
            # A cut-off output can end in a lone backslash or half an escape.
            body = re.sub(r"\\(?:u[0-9a-fA-F]{0,3})?$", "", body)
        try:
            return json.loads(f'"{body}"').strip()
        except json.JSONDecodeError:
            return body.strip()
