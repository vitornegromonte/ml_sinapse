#!/usr/bin/env python3
"""Gera pares candidatos de BO (via top-K por embedding + overlap de
entidades fortes) e enfileira em linkage:tasks para o
LinkageScoringWorker calcular o score real de cada par.

Dois geradores de candidato independentes, unidos e deduplicados antes de
enfileirar:

  1. Top-K por similaridade de embedding (índice HNSW já existente em
     gold.bo_embedding) — pega conexão semântica mesmo sem nenhuma entidade
     estruturada em comum.
  2. Overlap de entidades "fortes" (PESSOA_NOME, DOCUMENTO, PLACA, TELEFONE,
     etc. — mesma lista ENTIDADES_ID do motor original) — pega conexão
     exata mesmo quando o texto é semanticamente distante (BOs de naturezas
     diferentes que citam a mesma placa, por exemplo).

Sem isso, comparar todos os pares seria O(n²) — ~55M pares pros 10.518 BOs
de hoje, inviável rodar scoring completo (7 features + classificador) em
cada um. Normalização de entidade acontece em Python (mesma lógica de
motor_linkage.py/linkage_scoring_worker.py) — Postgres não tem a extensão
unaccent instalada, replicar a normalização em SQL puro seria frágil.

Idempotente: LinkageScoringWorker normaliza o par (menor id = origem, ver
CHECK id_bo_origem < id_bo_destino) e faz supersede-then-insert por par, uma
mensagem duplicada só recalcula o mesmo score de novo, sem duplicar linha.

Uso:
    python scripts/backfill_linkage_scores.py [--top-k 20] [--max-group-size 100] [--dry-run] [--limit N]

Requer:
    ML_DATABASE_URL     — leitura de gold.bo_embedding / gold.bo_entidades
    REDIS_URL           — para publicar em linkage:tasks
"""
import argparse
import asyncio
import logging
import re
import sys
import unicodedata
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings  # noqa: E402
from services.database import create_db_engine, create_session_factory  # noqa: E402
from services.linkage_candidates import pares_por_grupo  # noqa: E402
from services.task_publisher import enqueue, wait_for_capacity  # noqa: E402
from sqlalchemy import text as sql_text  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger("backfill_linkage_scores")

TOP_K_DEFAULT = 20

# A value shared by more BOs than this is a generic label, not an identifier —
# expanding it is the O(k²) blow-up this cap exists to stop. See
# services/linkage_candidates.py.
MAX_GROUP_SIZE_DEFAULT = 100

# Mesma lista de motor_linkage.py / linkage_scoring_worker.py — não editar
# sem editar as duas em conjunto.
ENTIDADES_ID = {
    "PESSOA_NOME", "DOCUMENTO", "PLACA", "BO_ID",
    "EMAIL", "TELEFONE", "INSTAGRAM", "URL",
    "CONTA_BANCARIA", "CHAVE_PIX", "OBJETO_VEICULO",
}
_TIPOS_PESSOA = {"PESSOA_NOME"}
_TIPOS_IDENTIFICADOR_DURO = {
    "DOCUMENTO", "PLACA", "BO_ID", "TELEFONE",
    "CONTA_BANCARIA", "AGENCIA_BANCARIA", "CHAVE_PIX",
}


def _normalizar_nome(nome: str) -> str:
    nome = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode()
    return " ".join(nome.upper().split())


def _normalizar_valor_entidade(texto: str, tipo: str) -> str:
    t = tipo.upper()
    if t in _TIPOS_PESSOA:
        return _normalizar_nome(texto)
    if t == "EMAIL":
        return texto.strip().lower()
    if t == "URL":
        return texto.strip().lower().rstrip("/")
    if t == "INSTAGRAM":
        return texto.strip().lower().lstrip("@")
    if t in _TIPOS_IDENTIFICADOR_DURO:
        return re.sub(r"[^A-Z0-9]", "", texto.upper())
    txt = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    txt = re.sub(r"[^\w\s]", " ", txt.lower())
    return " ".join(txt.split())


async def _candidatos_por_embedding(db, top_k: int, limit: int | None) -> set[tuple[int, int]]:
    query = sql_text(
        """
        SELECT e1.boletim_id, vizinhos.boletim_id
        FROM gold.bo_embedding e1, LATERAL (
            SELECT e2.boletim_id
            FROM gold.bo_embedding e2
            WHERE e2.is_current AND e2.boletim_id != e1.boletim_id
            ORDER BY e2.embedding <=> e1.embedding
            LIMIT :top_k
        ) vizinhos
        WHERE e1.is_current
        """
        + (" AND e1.boletim_id IN (SELECT boletim_id FROM gold.bo_embedding WHERE is_current ORDER BY boletim_id LIMIT :limit)" if limit else "")
    )
    params = {"top_k": top_k}
    if limit:
        params["limit"] = limit
    result = await db.execute(query, params)
    pares = set()
    for a, b in result.fetchall():
        pares.add((a, b) if a < b else (b, a))
    return pares


async def _candidatos_por_entidade_forte(db, max_group_size: int) -> set[tuple[int, int]]:
    labels = ", ".join(f"'{l}'" for l in ENTIDADES_ID)
    query = sql_text(
        f"""
        SELECT boletim_id, e->>'label' AS label, e->>'text' AS texto
        FROM gold.bo_entidades, jsonb_array_elements(entidades) e
        WHERE is_current AND e->>'label' IN ({labels})
        """
    )
    result = await db.execute(query)

    # {(label, valor_normalizado): {boletim_id, ...}}
    grupos: dict[tuple[str, str], set[int]] = {}
    for boletim_id, label, texto in result.fetchall():
        valor = _normalizar_valor_entidade(texto, label)
        if not valor:
            continue
        grupos.setdefault((label, valor), set()).add(boletim_id)

    pares, grupos_pulados = pares_por_grupo(grupos, max_group_size)
    if grupos_pulados:
        logger.warning(
            "  %d valor(es) genérico(s) ignorado(s): mais de %d BOs no mesmo "
            "valor, expandir seria O(k²).",
            grupos_pulados,
            max_group_size,
        )
    return pares


async def backfill(top_k: int, limit: int | None, max_group_size: int, dry_run: bool) -> None:
    engine = create_db_engine()
    session_factory = create_session_factory(engine)
    async with session_factory() as db:
        logger.info("Buscando candidatos por embedding (top-%d)...", top_k)
        pares_embedding = await _candidatos_por_embedding(db, top_k, limit)
        logger.info("  %d pares via embedding", len(pares_embedding))

        logger.info("Buscando candidatos por entidade forte...")
        pares_entidade = await _candidatos_por_entidade_forte(db, max_group_size)
        logger.info("  %d pares via entidade forte", len(pares_entidade))

    pares = pares_embedding | pares_entidade
    logger.info("Total de pares únicos (união, deduplicado): %d", len(pares))

    if not pares:
        logger.info("Nada a fazer.")
        return

    if dry_run:
        logger.info("DRY RUN — nada será publicado em %s.", settings.linkage_stream)
        for par in list(pares)[:5]:
            logger.info("  exemplo: %s", par)
        return

    enfileirados = 0
    for id_a, id_b in tqdm(pares, desc="Enfileirando", unit="par"):
        if enfileirados % 500 == 0:
            # Don't outrun the worker: the task stream is never trimmed, so
            # without this the whole backlog would sit in Redis RAM.
            await wait_for_capacity(settings.linkage_stream)
        await enqueue(settings.linkage_stream, {
            "id_bo_origem": id_a,
            "id_bo_destino": id_b,
        })
        enfileirados += 1

    logger.info("=" * 60)
    logger.info("RESUMO")
    logger.info("  Pares via embedding:      %d", len(pares_embedding))
    logger.info("  Pares via entidade forte: %d", len(pares_entidade))
    logger.info("  Pares únicos enfileirados: %d", enfileirados)
    logger.info("  Stream:                    %s", settings.linkage_stream)
    logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=TOP_K_DEFAULT)
    parser.add_argument("--limit", type=int, default=None, help="Limita a N BOs de origem (só afeta o gerador por embedding)")
    parser.add_argument(
        "--max-group-size",
        type=int,
        default=MAX_GROUP_SIZE_DEFAULT,
        help="Ignora valores de entidade compartilhados por mais de N BOs (genéricos: viram O(k²) pares)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    asyncio.run(backfill(args.top_k, args.limit, args.max_group_size, args.dry_run))


if __name__ == "__main__":
    main()
