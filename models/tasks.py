"""Typed payloads for the non-OCR workers, and the reject policy they share.

Each worker used to enforce its own contract, differently: ner_worker indexed
`data["text"]` (KeyError → retry → DLQ), llm_worker raised on a missing
document_id but logged-and-returned on empty text, embedding_worker and
linkage_scoring_worker logged-and-returned on missing ids. The same malformed
message dead-lettered on one stream and was silently discarded on another.

Policy, now uniform:

  * A payload that does not validate raises (pydantic.ValidationError, a
    ValueError subclass), so it takes the worker's existing retry → DLQ path.
    A malformed message is a producer bug and must be visible, not dropped.
  * A payload that validates but has nothing to do — empty text, a self-pair,
    a BO whose silver row isn't ready — is skipped by the worker with a
    warning. That is data state, and retrying it forever would not help.

OcrTask lives in models/ocr/task.py, next to the model contract its fields
describe (image_url/image_base64/image_path/pdf_bucket+pdf_key).
"""
from pydantic import BaseModel, Field


class NerTask(BaseModel):
    document_id: str = Field(default="", description="Identifier for the source document")
    text: str = Field(description="Text to run NER over")


class LlmTask(BaseModel):
    document_id: str = Field(description="Identifier for the source document")
    # None/empty is a skip, not an error: OCR can legitimately produce no text.
    text: str | None = Field(default=None, description="Text to summarize")


class EmbeddingTask(BaseModel):
    boletim_id: int = Field(description="silver.boletim.id_infopol to embed")


class LinkageTask(BaseModel):
    id_bo_origem: int = Field(description="First BO of the candidate pair")
    id_bo_destino: int = Field(description="Second BO of the candidate pair")
    pipeline_run_id: str | None = Field(
        default=None, description="Correlation id of the run that produced the pair"
    )


class ScoringTask(BaseModel):
    boletim_id: int = Field(description="silver.boletim.id_infopol to score")
    ato_infracional: str = Field(
        description="Natureza whose weights/penalty apply (see occurrence_scoring)"
    )
    # Which maturity features are present. Deriving them from the BO text is a
    # separate step; this task only scores what it is told. A missing feature
    # counts as absent, so extra or unknown flags are harmless.
    features: dict[str, bool] = Field(default_factory=dict)
    pipeline_run_id: str | None = Field(
        default=None, description="Correlation id of the run that produced the BO"
    )
