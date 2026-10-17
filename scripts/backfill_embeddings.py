#!/usr/bin/env python3
"""Enfileira em embedding:tasks todos os Boletins que ainda não têm um
embedding corrente em gold.bo_embedding — o backfill inicial para popular a
tabela que EmbeddingWorker consome (ver workers/embedding_worker.py).

Não escreve no banco diretamente: só lê silver.boletim (texto) e
gold.bo_embedding (para saber o que já tem embedding) e publica mensagens em
embedding:tasks. Quem grava é o EmbeddingWorker já rodando do outro lado —
mesma separação producer/consumer que o resto do pipeline usa (ver
services/task_publisher.py).

Idempotente por natureza: um boletim_id que já tem uma linha is_current=true
em gold.bo_embedding é pulado. Rodar de novo depois de uma falha parcial (ou
depois de novos BOs chegarem) é seguro — só processa o que falta.

Uso:
    python scripts/backfill_embeddings.py [--limit N] [--dry-run]

Requer:
    ML_DATABASE_URL     — leitura de silver.boletim / gold.bo_embedding
                          (role sinapse_ml, que lê ambos os schemas)
    REDIS_URL           — para publicar em embedding:tasks
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings  # noqa: E402
from services.database import create_db_engine, create_session_factory  # noqa: E402
from services.task_publisher import enqueue  # noqa: E402
from sqlalchemy import text as sql_text  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger("backfill_embeddings")


async def _fetch_pending_boletins(db, limit: int | None) -> list[dict]:
    """Boletins com narrativa não vazia e sem embedding corrente ainda."""
    query = sql_text(
        """
        SELECT b.id_infopol, b.narrativa_normalizada
        FROM silver.boletim b
        LEFT JOIN gold.bo_embedding e
            ON e.boletim_id = b.id_infopol AND e.is_current = true
        WHERE e.id IS NULL
          AND b.narrativa_normalizada IS NOT NULL
          AND b.narrativa_normalizada != ''
        ORDER BY b.id_infopol
        """
        + (" LIMIT :limit" if limit else "")
    )
    params = {"limit": limit} if limit else {}
    result = await db.execute(query, params)
    return [{"boletim_id": row[0], "text": row[1]} for row in result.fetchall()]


async def backfill(limit: int | None, dry_run: bool) -> None:
    engine = create_db_engine()
    session_factory = create_session_factory(engine)
    async with session_factory() as db:
        pending = await _fetch_pending_boletins(db, limit)

    logger.info("Boletins pendentes de embedding: %d", len(pending))
    if not pending:
        logger.info("Nada a fazer.")
        return

    if dry_run:
        logger.info("DRY RUN — nada será publicado em %s.", settings.embedding_stream)
        for item in pending[:5]:
            logger.info("  exemplo: boletim_id=%s", item["boletim_id"])
        return

    enfileirados = 0
    for item in tqdm(pending, desc="Enfileirando", unit="BO"):
        await enqueue(settings.embedding_stream, {
            "boletim_id": item["boletim_id"],
            "text": item["text"],
        })
        enfileirados += 1

    logger.info("=" * 60)
    logger.info("RESUMO")
    logger.info("  Boletins enfileirados: %d", enfileirados)
    logger.info("  Stream:                %s", settings.embedding_stream)
    logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    asyncio.run(backfill(args.limit, args.dry_run))


if __name__ == "__main__":
    main()
