"""Consumes embedding:tasks, encodes BO narrative text, and writes the
resulting vector to gold.bo_embedding.

The narrative is read from silver.boletim by boletim_id: the task message
carries only the id, so the backfill backlog stays small in Redis (see
services/task_publisher.py) instead of shipping KBs of text per task.

Unlike ocr_worker/ner_worker (which only log or publish a result for a
downstream consumer to apply), this worker writes Gold directly — sinapse_ml
already has write grants there, and gold.bo_embedding is exactly the kind of
ML artifact Gold exists for (see app/modules/gold/models.py). No backend-side
consumer needed, unlike OCR's silver write (blocked by role grants — see
ocr_result_consumer.py).

First worker to touch Postgres beyond read — see
services/database.py's docstring ("Per-worker Gold table models get defined
here as each worker's DB writes get built out"). Deliberately raw SQL via
text(), not the backend's SQLAlchemy ORM models: this package and backend are
separate deployables that shouldn't share a Python package boundary (same
rationale ocr_result_consumer.py gives for not reusing BaseWorker).
"""
import asyncio
import json

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from models.embedding_model import SentenceTransformerEncoder
from workers.base_worker import BaseWorker

MODEL_NAME = "bo_linkage_encoder"
MODEL_VERSION = "neuralmind/bert-base-portuguese-cased"


class EmbeddingWorker(BaseWorker):
    def __init__(self, cfg=settings):
        super().__init__(cfg.embedding_stream, cfg)
        self._encoder: SentenceTransformerEncoder | None = None
        self._model_id: int | None = None

    def _load_model(self) -> None:
        device = (
            f"cuda:{self.cfg.embedding_gpu_device}"
            if self._cuda_available()
            else "cpu"
        )
        self._encoder = SentenceTransformerEncoder(self.cfg.embedding_model_name, device=device)
        self._encoder.load()

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    async def _get_or_create_model_id(self, db: AsyncSession) -> int:
        """Mirrors import_bo_ner.py's get_or_create_model_registry pattern —
        one row per (name, version), reused across every embedding this
        worker writes. Resolved once and cached on the instance, not
        re-queried per message.

        Must be called from inside an already-open `async with db.begin()`
        block (see process()) — this method does NOT commit itself, so its
        INSERT lands in whatever transaction the caller already started."""
        if self._model_id is not None:
            return self._model_id

        result = await db.execute(
            text(
                "SELECT id FROM gold.model_registry WHERE name = :name AND version = :version"
            ),
            {"name": MODEL_NAME, "version": MODEL_VERSION},
        )
        row = result.first()
        if row is not None:
            self._model_id = row[0]
            return self._model_id

        result = await db.execute(
            text(
                """
                INSERT INTO gold.model_registry (name, version, stage, hyperparameters)
                VALUES (:name, :version, 'production', :hyperparameters)
                RETURNING id
                """
            ),
            {
                "name": MODEL_NAME,
                "version": MODEL_VERSION,
                "hyperparameters": json.dumps({"library": "sentence-transformers"}),
            },
        )
        self._model_id = result.scalar_one()
        return self._model_id

    async def _load_narrativa(self, db: AsyncSession, boletim_id: int) -> str | None:
        result = await db.execute(
            text(
                "SELECT narrativa_normalizada FROM silver.boletim "
                "WHERE id_infopol = :id"
            ),
            {"id": boletim_id},
        )
        row = result.first()
        return row[0] if row is not None else None

    async def process(self, data: dict) -> None:
        boletim_id = data.get("boletim_id")
        if not boletim_id:
            self.logger.warning(
                "Skipping message with missing boletim_id: %s", data
            )
            return

        boletim_id = int(boletim_id)

        async with self.db_session_factory() as db:  # type: AsyncSession
            text_content = await self._load_narrativa(db, boletim_id)

        if not text_content or not text_content.strip():
            self.logger.warning(
                "Skipping boletim_id=%s — no narrative text in silver.boletim",
                boletim_id,
            )
            return

        if self._encoder is None:
            await asyncio.to_thread(self._load_model)

        vec = await asyncio.to_thread(self._encoder.encode, [text_content])
        embedding = vec[0].tolist()

        async with self.db_session_factory() as db:  # type: AsyncSession
            # Atomic supersede-then-insert, per GoldMixin's contract (see
            # app/core/mixins.py docstring): the old current row (if any)
            # gets superseded, the new one becomes current, both in one
            # transaction — never a window with zero or two current rows.
            # Everything (including model_id resolution) happens inside this
            # one begin() block: AsyncSession autobegins on first use, so any
            # prior statement on `db` (e.g. a SELECT run before this block)
            # would already have an implicit transaction open, and this
            # explicit begin() would then conflict with it.
            async with db.begin():
                model_id = await self._get_or_create_model_id(db)

                await db.execute(
                    text(
                        """
                        UPDATE gold.bo_embedding
                        SET is_current = false, superseded_at = now()
                        WHERE boletim_id = :boletim_id AND is_current = true
                        """
                    ),
                    {"boletim_id": boletim_id},
                )
                await db.execute(
                    text(
                        """
                        INSERT INTO gold.bo_embedding
                            (boletim_id, embedding, model_id, generated_at, pipeline_run_id)
                        VALUES
                            (:boletim_id, CAST(:embedding AS vector), :model_id, now(), :pipeline_run_id)
                        """
                    ),
                    {
                        "boletim_id": boletim_id,
                        "embedding": str(embedding),
                        "model_id": model_id,
                        "pipeline_run_id": data.get("pipeline_run_id"),
                    },
                )

        self.logger.info(
            "Embedding written: boletim_id=%s dim=%d", boletim_id, len(embedding)
        )

    def run(self) -> None:
        try:
            super().run()
        finally:
            if self._encoder is not None:
                self._encoder.unload()
            self.logger.info("Embedding worker finished")
