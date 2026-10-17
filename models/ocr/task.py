from pydantic import BaseModel, Field


class OcrTask(BaseModel):
    document_id: str = Field(default="", description="Identifier for the source document")
    image_url: str | None = Field(default=None, description="Public URL of the image")
    image_base64: str | None = Field(default=None, description="Base64-encoded image data")
    image_path: str | None = Field(default=None, description="Local filesystem path to image")
    # PDF support — mutually exclusive with the image_* fields above. The
    # worker downloads pdf_key from pdf_bucket (MinIO), splits it into text +
    # image chunks (PdfExtractor), and OCRs each image chunk in turn — see
    # OcrWorker._process_pdf. One task still means one PDF processed serially
    # by one worker; horizontal scale comes from multiple *different*
    # documents being picked up by different worker replicas (existing
    # consumer-group behavior), not from splitting pages across workers.
    pdf_bucket: str | None = Field(default=None, description="MinIO bucket containing the PDF")
    pdf_key: str | None = Field(default=None, description="Object key of the PDF within pdf_bucket")

    def has_image(self) -> bool:
        return bool(self.image_url or self.image_base64 or self.image_path)

    def has_pdf(self) -> bool:
        return bool(self.pdf_bucket and self.pdf_key)
