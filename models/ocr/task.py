"""The payload of an OCR task, as the backend publishes it to `ocr:tasks`."""
from pydantic import BaseModel, Field


class OcrTask(BaseModel):
    """One document to OCR: a single image, or a PDF stored in MinIO.

    Set exactly one source. The image fields (`image_url`, `image_base64`,
    `image_path`) and the PDF fields (`pdf_bucket` with `pdf_key`) exclude each
    other. If both are set, the PDF wins (see OcrWorker.process).
    """

    document_id: str = Field(default="", description="Identifier for the source document")

    # Image source: one of these.
    image_url: str | None = Field(default=None, description="Public URL of the image")
    image_base64: str | None = Field(default=None, description="Base64-encoded image data")
    image_path: str | None = Field(default=None, description="Local filesystem path to image")

    # PDF source: both fields. The worker downloads `pdf_key` from `pdf_bucket`
    # (MinIO) and OCRs the scanned parts. One task is one PDF, handled by one
    # worker. More throughput comes from more worker replicas, each taking
    # other documents from the consumer group.
    pdf_bucket: str | None = Field(default=None, description="MinIO bucket containing the PDF")
    pdf_key: str | None = Field(default=None, description="Object key of the PDF within pdf_bucket")

    def has_image(self) -> bool:
        return bool(self.image_url or self.image_base64 or self.image_path)

    def has_pdf(self) -> bool:
        return bool(self.pdf_bucket and self.pdf_key)
