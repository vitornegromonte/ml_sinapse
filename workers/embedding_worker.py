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

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from models.embedding_model import SentenceTransformerEncoder
from models.tasks import EmbeddingTask
from services.gold import get_or_create_model_id
from workers.base_worker import BaseWorker

MODEL_NAME = "bo_linkage_encoder"
MODEL_VERSION = "neuralmind/bert-base-portuguese-cased"


class EmbeddingWorker(BaseWorker):
    supports_batch = True

    def __init__(self, cfg=settings):
        super().__init__(cfg.embedding_stream, cfg)
        self._encoder: SentenceTransformerEncoder | None = None

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

    async def _load_narrativas(
        self, db: AsyncSession, boletim_ids: list[int]
    ) -> dict[int, str | None]:
        result = await db.execute(
            text(
                "SELECT id_infopol, narrativa_normalizada FROM silver.boletim "
                "WHERE id_infopol = ANY(:ids)"
            ),
            {"ids": boletim_ids},
        )
        return {row[0]: row[1] for row in result}

    async def process(self, data: dict) -> None:
        (error,) = await self.process_batch([data])
        if error is not None:
            raise error

    async def process_batch(self, items: list[dict]) -> list[Exception | None]:
        """One SELECT, one encode() and one transaction for the whole batch.

        The encode is the dominant cost and scales ~10x with batch size; the
        three round trips per task (SELECT, UPDATE, INSERT) become three for
        the batch. A malformed payload or a BO with no narrative is settled on
        its own, as in the single-item path.
        """
        errors: list[Exception | None] = [None] * len(items)
        tasks: list[tuple[int, int, dict]] = []
        for position, data in enumerate(items):
            try:
                tasks.append((position, EmbeddingTask(**data).boletim_id, data))
            except Exception as exc:
                errors[position] = exc
        if not tasks:
            return errors

        async with self.db_session_factory() as db:  # type: AsyncSession
            narratives = await self._load_narrativas(db, [t[1] for t in tasks])

        ready: dict[int, tuple[int, int, dict, str]] = {}
        for position, boletim_id, data in tasks:
            text_content = narratives.get(boletim_id)
            if not text_content or not text_content.strip():
                self.logger.warning(
                    "Skipping boletim_id=%s — no narrative text in silver.boletim",
                    boletim_id,
                )
                continue
            # A batch can repeat a BO; keep only its last appearance. The one
            # UPDATE cannot supersede a row an earlier INSERT of this same
            # batch created, and two current rows break GoldMixin's contract.
            ready[boletim_id] = (position, boletim_id, data, text_content)
        if not ready:
            return errors
        batch = list(ready.values())

        if self._encoder is None:
            await asyncio.to_thread(self._load_model)

        try:
            vecs = await asyncio.to_thread(
                self._encoder.encode, [item[3] for item in batch]
            )
        except Exception as exc:
            for position, *_ in batch:
                errors[position] = exc
            return errors

        try:
            async with self.db_session_factory() as db:  # type: AsyncSession
                # Atomic supersede-then-insert, per GoldMixin's contract (see
                # app/core/mixins.py docstring): the old current row (if any)
                # gets superseded, the new one becomes current, both in one
                # transaction — never a window with zero or two current rows.
                async with db.begin():
                    model_id = await get_or_create_model_id(
                        db,
                        MODEL_NAME,
                        MODEL_VERSION,
                        {"library": "sentence-transformers"},
                    )

                    # The "never two current rows" guarantee depends on the
                    # backend's partial unique index on gold.bo_embedding
                    # (boletim_id) WHERE is_current. The index lives in the
                    # sinapse backend Alembic migrations, not in this repo.
                    await db.execute(
                        text(
                            """
                            UPDATE gold.bo_embedding
                            SET is_current = false, superseded_at = now()
                            WHERE boletim_id = ANY(:ids) AND is_current = true
                            """
                        ),
                        {"ids": [item[1] for item in batch]},
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
                        [
                            {
                                "boletim_id": boletim_id,
                                "embedding": str(vec.tolist()),
                                "model_id": model_id,
                                "pipeline_run_id": data.get("pipeline_run_id"),
                            }
                            for (_, boletim_id, data, _), vec in zip(batch, vecs)
                        ],
                    )
        except Exception as exc:
            for position, *_ in batch:
                errors[position] = exc
            return errors

        self.logger.info(
            "Embeddings written: %d of %d tasks", len(batch), len(items)
        )
        return errors

    def run(self) -> None:
        try:
            super().run()
        finally:
            if self._encoder is not None:
                self._encoder.unload()
            self.logger.info("Embedding worker finished")
