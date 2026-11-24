"""Prompts sent to the OCR models.

They are in Portuguese on purpose: the documents are Brazilian police reports
(BOs). Each model family keeps the user prompt it was tuned with. The system
prompt is shared.
"""

SYSTEM_PROMPT = (
    "Você é um assistente de transcrição de documentos, "
    "sua função é extrair todo o texto de uma imagem"
)

# Dharma-OCR (Qwen2.5-VL): asks for the text and every table on the page.
QWEN_USER_PROMPT = (
    "Extraia todo o texto e todas as tabelas deste documento. "
    "Transcreva todas as tabelas caso existam multiplas."
)

# GLM-OCR: asks for a plain transcription in Brazilian Portuguese.
GLM_USER_PROMPT = (
    "Transcreva todo o texto presente nesta imagem com extrema "
    "precisão, no idioma português do Brasil."
)
