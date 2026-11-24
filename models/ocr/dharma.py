"""Dharma-OCR-LITE, the default OCR model."""
import json
import logging
import re

from models.ocr import prompts
from models.ocr.vllm_chat_base import VllmChatOcrModel

logger = logging.getLogger(__name__)

# A ```json ... ``` block around the whole answer.
_CODE_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
# The start of the "text" string inside a JSON answer: "text": "
_TEXT_FIELD_START = re.compile(r'"text"\s*:\s*"')


class DharmaOcrModel(VllmChatOcrModel):
    """`Dharma-AI/Dharma-OCR-LITE`, a Qwen2.5-VL fine-tune, on vLLM.

    Output format: the model answers each page with a JSON object,
    `{"header": ..., "margin": ..., "footer": ..., "text": "..."}`, with null
    for regions the page does not have. `predict()` and `predict_batch()`
    return only the `text` field, as plain text. `header`, `margin` and
    `footer` are dropped as page noise.

    Quality: it scored lowest of 8 models on the Portuguese MDPBench samples
    (Overall 56.0, GLM-OCR 85.1) and was chosen anyway. Measure it on real BOs.
    See docs/adr-004-multimodal-ocr.md.

    Cost: Qwen2.5-VL reads an image at its native size, so `ocr_render_dpi`
    sets the number of image tokens.
    """

    SYSTEM_PROMPT = prompts.SYSTEM_PROMPT
    USER_PROMPT = prompts.QWEN_USER_PROMPT

    # Qwen2.5-VL also accepts video. Turning it off removes vLLM's memory
    # profiling for a maximum-size video. That run needed about 20 GiB at the
    # native 128k context, which made the 15% GPU cap impossible to meet.
    MM_LIMITS = {"image": 1, "video": 0}

    # The model's own generation_config.json sets 1.05. Our explicit
    # SamplingParams ignores that file. Without it, 2 of 5 image chunks of a
    # test PDF looped until max_tokens (see docs/adr-012).
    REPETITION_PENALTY = 1.05

    def _postprocess(self, text: str) -> str:
        """Return the body text of the model's JSON answer.

        Cases, in order:
          1. A fenced ```json block is unwrapped first.
          2. Valid JSON with a "text" field gives that field. A null or
             non-string "text" gives "".
          3. JSON that does not parse, for example because max_tokens cut it
             off, gives the part of the "text" string that was written.
          4. Anything else is plain text and is returned as it is.
        Page content is never logged.
        """
        raw = text.strip()
        fenced = _CODE_FENCE.match(raw)
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
        """Read the "text" string out of JSON that does not parse as a whole.

        Reads up to the closing quote, or to the end when the answer was cut
        off. Without a `"text":` field the answer is plain text, so return it.
        """
        match = _TEXT_FIELD_START.search(raw)
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
            # A cut-off answer can end in a lone backslash or half an escape
            # such as \u00e. Drop it so the rest still decodes.
            body = re.sub(r"\\(?:u[0-9a-fA-F]{0,3})?$", "", body)

        try:
            return json.loads(f'"{body}"').strip()
        except json.JSONDecodeError:
            return body.strip()
