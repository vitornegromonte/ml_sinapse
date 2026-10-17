import asyncio

import spacy
from spacy import Language

from config import settings
from models.base_model import BaseModelWrapper
from services.model_sync import sync_if_missing
from workers.base_worker import BaseWorker


class NerWorker(BaseWorker):
    def __init__(self, cfg=settings):
        super().__init__(cfg.ner_stream, cfg)
        self._nlp: Language | None = None

    def _load_model(self) -> None:
        uri = self.cfg.ner_model_uri
        if uri:
            self.logger.info("Loading NER model from MLflow: %s", uri)
            import mlflow.spacy

            self._nlp = mlflow.spacy.load_model(uri)
            self.logger.info("NER model loaded from MLflow")
            return

        path = self.cfg.ner_model_path
        if self.cfg.ner_model_minio_prefix:
            sync_if_missing(
                local_dir=path,
                bucket=self.cfg.minio_bucket_models,
                prefix=self.cfg.ner_model_minio_prefix,
                endpoint=self.cfg.minio_endpoint,
                access_key=self.cfg.minio_access_key,
                secret_key=self.cfg.minio_secret_key,
            )

        self.logger.info("Loading spaCy model from %s", path)
        self._nlp = spacy.load(path)
        self.logger.info("spaCy model loaded")

    async def process(self, data: dict) -> None:
        if self._nlp is None:
            await asyncio.to_thread(self._load_model)

        text = data["text"]

        def run_ner(text: str) -> list[dict]:
            doc = self._nlp(text)
            return [
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                }
                for ent in doc.ents
            ]

        entities = await asyncio.to_thread(run_ner, text)

        self.logger.info(
            "NER result: document_id=%s entity_count=%d entities=%s",
            data.get("document_id", "?"),
            len(entities),
            entities,
        )
