#!/usr/bin/env python3
"""Importa Boletins de Ocorrência + entidades NER (gemma4_ner) para o SINAPSE.

Duas fontes, ligadas pelo bo_id/doc_id em comum:
  - bos_reconstruidos.jsonl    BO bruto: data_fato, hora_fato, data_registro,
                                delegacia, historico, ... (tem os campos que
                                Boletim exige e o JSONL de saída do NER não tem)
  - annotated_documents.jsonl  saída do NER: doc_id, text, entities (uma linha
                                por BO com resultado — 10.518 de 11.838 no
                                bos_reconstruidos.jsonl bruto; o resto ficou de
                                fora por historico vazio)

silver.boletim e gold.model_registry estão vazios agora — não existe pipeline
de ingestão de Boletim ainda (só o schema). Este script cria os dois:

  1. gold.model_registry  — 1 linha (gemma4_ner), criada se ausente
  2. silver.boletim       — 1 linha por BO com resultado NER, idempotente por
                            bo_num (id_infopol é derivado de bo_num — ver
                            derive_id_infopol — não existe no JSONL um id
                            Infopol de verdade, mesma situação de
                            inquerito_extraction.derive_id_spp)
  3. gold.bo_entidades    — 1 linha por BO, idempotente por
                            (boletim_id, pipeline_run_id)

Processamento em lotes: cada lote grava em DUAS transações, porque usa dois
papéis: silver (`AsyncSessionLocal`, sinapse_backend) commita primeiro, e
depois gold (`MlSessionLocal`, sinapse_ml). Se a fase gold falhar depois do
commit da silver, o lote é logado com os doc_ids e pulado; re-rodar o script
depois é seguro (tudo aqui é idempotente) e repara a parte gold. Não há rollback
cruzando as duas transações.

Uso:
    python import_bo_ner.py \
        --bos-jsonl path/to/bos_reconstruidos.jsonl \
        --ner-jsonl path/to/annotated_documents.jsonl \
        --pipeline-run-id bos_reconstruidos_gemma4_e4b_preannotate_20260729_001 \
        [--batch-size 200] [--limit N] [--dry-run]

Requer APP_DATABASE_URL (mesma convenção dos outros scripts de import da raiz
do repo sinapse) apontando pro Postgres alcançável de onde o script roda.

DEPENDÊNCIA: este script usa o ORM do backend (app.*). Ele espera o checkout
do repo sinapse como irmão deste (../sinapse/backend) — não faz parte do
standalone ml_sinapse. Uso:
    python scripts/import_bo_ner.py ...   # com ../sinapse/backend presente
"""
import argparse
import asyncio
import hashlib
import json
import logging
import sys
import time
import zlib
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from tqdm import tqdm

# Backend ORM lives in the sibling sinapse repo (see dependency note above).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "sinapse" / "backend"))

import app.models  # noqa: E402,F401 — registers every ORM model before mapper use
from app.core.database import AsyncSessionLocal  # noqa: E402 — sinapse_backend: writes silver
from app.core.repository import BaseRepository  # noqa: E402
from app.modules.boletins.models import Boletim  # noqa: E402
from app.modules.gold.models import BOEntidades, ModelRegistry  # noqa: E402
from app.modules.unidades.models import Unidade  # noqa: E402

# sinapse_backend can write silver but NOT gold, and sinapse_ml is the exact
# reverse (read silver, write gold — never silver) — this is the same role
# split ocr_worker/ocr_result_consumer.py already respects for OCR: the ML
# side never writes silver.inquerito directly. Found live: get_or_create_
# model_registry() failed with InsufficientPrivilegeError trying to INSERT
# gold.model_registry through the sinapse_backend connection. Boletim/Unidade
# go through AsyncSessionLocal (silver); ModelRegistry/BOEntidades go through
# this second engine instead.
import os  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession  # noqa: E402

_ML_DATABASE_URL = os.getenv(
    "ML_DATABASE_URL",
    "postgresql://sinapse_ml:change_me_ml_worker@localhost:5433/sinapse_db",
).replace("postgresql://", "postgresql+asyncpg://", 1)
_ml_engine = create_async_engine(_ML_DATABASE_URL, echo=False)
MlSessionLocal = async_sessionmaker(bind=_ml_engine, class_=AsyncSession, expire_on_commit=False)

logger = logging.getLogger("import_bo_ner")

MODEL_NAME = "gemma4_ner"
MODEL_VERSION = "google/gemma-4-E4B-it"


def setup_logging(log_file: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def derive_id_infopol(bo_num: str) -> int:
    """id_infopol (PK real de Boletim) não existe no JSONL — só o bo_num
    (ex. "21F2058000746") existe. CRC32 mascarado pra 31 bits: determinístico
    (mesmo bo_num -> mesmo id sempre, essencial pra idempotência entre
    execuções), cabe garantido num INTEGER de 4 bytes, e a chance de colisão
    em ~11 mil registros num espaço de 2^31 é desprezível. Mesma situação de
    inquerito_extraction.derive_id_spp — ajustar se/quando o id Infopol real
    ficar disponível."""
    return zlib.crc32(bo_num.encode("utf-8")) & 0x7FFFFFFF


def parse_data_hora(data_str: str, hora_str: str | None) -> datetime:
    """"25/01/2021" [+ "08:00"] -> datetime aware UTC. Sem hora -> meia-noite."""
    dia, mes, ano = data_str.split("/")
    hora, minuto = (hora_str or "00:00").split(":")
    return datetime(int(ano), int(mes), int(dia), int(hora), int(minuto), tzinfo=timezone.utc)


def compute_dia_semana(data_str: str) -> int:
    """0=segunda...6=domingo — Python's date.weekday() já usa essa convenção
    (Monday=0), não precisa de ajuste."""
    dia, mes, ano = data_str.split("/")
    return date(int(ano), int(mes), int(dia)).weekday()


def load_bos_index(path: Path) -> dict[str, dict]:
    index: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            index[rec["bo_id"]] = rec
    return index


def count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


@dataclass
class Stats:
    boletins_criados: int = 0
    boletins_existentes: int = 0
    entidades_criadas: int = 0
    entidades_existentes: int = 0
    unidades_criadas: int = 0
    doc_id_sem_bo: int = 0
    lotes_falhos: int = 0
    registros_em_lotes_falhos: int = 0
    doc_ids_falhos: list = field(default_factory=list)

    def log_summary(self, elapsed_s: float, total: int) -> None:
        logger.info("=" * 60)
        logger.info("RESUMO")
        logger.info("  Boletins criados:        %d", self.boletins_criados)
        logger.info("  Boletins já existentes:   %d", self.boletins_existentes)
        logger.info("  Unidades criadas:         %d", self.unidades_criadas)
        logger.info("  BOEntidades criadas:      %d", self.entidades_criadas)
        logger.info("  BOEntidades já existentes:%d", self.entidades_existentes)
        logger.info("  doc_id sem BO correspond.:%d", self.doc_id_sem_bo)
        logger.info("  Lotes com falha:          %d (%d registros)", self.lotes_falhos, self.registros_em_lotes_falhos)
        if self.doc_ids_falhos:
            logger.info("  doc_ids em lotes falhos:  %s", ", ".join(self.doc_ids_falhos[:50]))
        logger.info("  Tempo total:              %.1fs (%.1f registros/s)", elapsed_s, total / elapsed_s if elapsed_s else 0)
        logger.info("=" * 60)


async def get_or_create_model_registry(stage: str, run_summary: dict, dry_run: bool) -> int:
    async with MlSessionLocal() as db:
        repo = BaseRepository(ModelRegistry, db)
        existing = await repo.filter_by(name=MODEL_NAME, version=MODEL_VERSION)
        if existing:
            logger.info("ModelRegistry já existe: %s/%s (id=%d)", MODEL_NAME, MODEL_VERSION, existing[0].id)
            return existing[0].id
        if dry_run:
            logger.info("[dry-run] criaria ModelRegistry: %s/%s", MODEL_NAME, MODEL_VERSION)
            return -1
        created = await repo.create({
            "name": MODEL_NAME,
            "version": MODEL_VERSION,
            "stage": stage,
            "hyperparameters": {
                "temperature": run_summary.get("temperature"),
                "max_tokens": run_summary.get("generation_config", {}).get("max_tokens"),
                "matching_policy": run_summary.get("matching_policy"),
                "conditions": run_summary.get("conditions"),
            },
            "metrics": {
                "num_docs": run_summary.get("num_docs"),
                "num_entity_types": run_summary.get("num_entity_types"),
                "num_parse_errors": run_summary.get("num_parse_errors"),
                "num_unmatched_entities": run_summary.get("num_unmatched_entities"),
            },
        })
        logger.info("ModelRegistry criado: %s/%s (id=%d)", MODEL_NAME, MODEL_VERSION, created.id)
        return created.id


async def _process_silver_phase(
    batch: list[dict],
    bos_index: dict[str, dict],
    unidade_cache: dict[str, int],
    dry_run: bool,
    stats: Stats,
) -> dict[str, int]:
    """Resolve/cria Unidade + Boletim (schema silver, via sinapse_backend).
    Retorna {doc_id: boletim_id} para os registros que puderam ser resolvidos
    — usado pela fase gold logo em seguida."""
    boletim_ids: dict[str, int] = {}
    async with AsyncSessionLocal() as db:
        unidade_repo = BaseRepository(Unidade, db)
        boletim_repo = BaseRepository(Boletim, db)

        async with db.begin():
            for ner_record in batch:
                doc_id = ner_record["doc_id"]
                bo_record = bos_index.get(doc_id)
                if bo_record is None:
                    logger.warning("doc_id=%s sem registro em bos_reconstruidos — pulado", doc_id)
                    stats.doc_id_sem_bo += 1
                    continue

                # --- Unidade (resolve ou cria, cache em memória por delegacia) ---
                delegacia = (bo_record.get("delegacia") or "Não informado").strip()
                unidade_id = unidade_cache.get(delegacia)
                if unidade_id is None:
                    existing_unidade = await unidade_repo.get_by_field("nome", delegacia)
                    if existing_unidade:
                        unidade_id = existing_unidade.id_unidade
                    elif dry_run:
                        unidade_id = -1
                        stats.unidades_criadas += 1
                    else:
                        new_unidade = Unidade(
                            nome=delegacia,
                            tipo="delegacia",
                            # Não temos o endereço da unidade em si, só do fato
                            # — município do fato como aproximação razoável
                            # (delegacias costumam processar fatos do próprio
                            # município, mas não é garantido).
                            logradouro="Não informado",
                            municipio=(bo_record.get("municipio") or "Não informado"),
                        )
                        db.add(new_unidade)
                        await db.flush()
                        unidade_id = new_unidade.id_unidade
                        stats.unidades_criadas += 1
                    unidade_cache[delegacia] = unidade_id

                # --- Boletim (resolve ou cria, idempotente por id_infopol derivado de bo_num) ---
                bo_num = bo_record["bo_id"]
                id_infopol = derive_id_infopol(bo_num)
                existing_boletim = await boletim_repo.get_by_id(id_infopol, include_inactive=True)
                if existing_boletim is not None:
                    stats.boletins_existentes += 1
                    boletim_ids[doc_id] = existing_boletim.id_infopol
                    continue

                if not dry_run:
                    historico = bo_record.get("historico") or ""
                    new_boletim = Boletim(
                        id_infopol=id_infopol,
                        bo_num=bo_num,
                        unidade_id=unidade_id if unidade_id != -1 else None,
                        horario_ocorrencia=parse_data_hora(bo_record["data_fato"], bo_record.get("hora_fato")),
                        horario_cadastro=parse_data_hora(bo_record["data_registro"], None),
                        dia_semana=compute_dia_semana(bo_record["data_fato"]),
                        # Mesmo texto nos dois campos — não há pipeline de
                        # normalização separado ainda (ver docstring de
                        # Boletim). narrativa_raw preserva o histórico
                        # exatamente como veio do reconstruído.
                        narrativa_raw=historico,
                        narrativa_normalizada=historico,
                        status="ner_complete",
                    )
                    db.add(new_boletim)
                    await db.flush()
                stats.boletins_criados += 1
                boletim_ids[doc_id] = id_infopol
        # commit ao sair do `async with db.begin()` sem exceção
    return boletim_ids


async def _process_gold_phase(
    batch: list[dict],
    boletim_ids: dict[str, int],
    model_id: int,
    pipeline_run_id: str,
    generated_at: datetime,
    dry_run: bool,
    stats: Stats,
) -> None:
    """Cria BOEntidades (schema gold, via sinapse_ml) para os registros que a
    fase silver conseguiu resolver — precisa rodar depois, boletim_id é FK.

    O contrato "no máximo uma linha is_current por BO" também depende do índice
    único parcial em gold.bo_entidades WHERE is_current, definido nas migrations
    Alembic do backend (sinapse/backend), não neste repo."""
    async with MlSessionLocal() as db:
        entidades_repo = BaseRepository(BOEntidades, db)

        async with db.begin():
            for ner_record in batch:
                doc_id = ner_record["doc_id"]
                boletim_id = boletim_ids.get(doc_id)
                if boletim_id is None:
                    continue  # já contado como doc_id_sem_bo na fase silver

                dup = await entidades_repo.filter_by(boletim_id=boletim_id, pipeline_run_id=pipeline_run_id)
                if dup:
                    stats.entidades_existentes += 1
                    continue

                if not dry_run:
                    new_entidades = BOEntidades(
                        boletim_id=boletim_id,
                        entidades=ner_record["entities"],
                        model_id=model_id,
                        generated_at=generated_at,
                        pipeline_run_id=pipeline_run_id,
                        source_hash=hashlib.sha256(ner_record["text"].encode("utf-8")).hexdigest(),
                    )
                    db.add(new_entidades)
                    await db.flush()
                stats.entidades_criadas += 1


async def process_batch(
    batch: list[dict],
    bos_index: dict[str, dict],
    model_id: int,
    pipeline_run_id: str,
    generated_at: datetime,
    unidade_cache: dict[str, int],
    dry_run: bool,
) -> Stats:
    stats = Stats()
    boletim_ids = await _process_silver_phase(batch, bos_index, unidade_cache, dry_run, stats)
    if model_id != -1:  # -1 == dry-run sentinel from get_or_create_model_registry
        await _process_gold_phase(batch, boletim_ids, model_id, pipeline_run_id, generated_at, dry_run, stats)
    elif dry_run:
        stats.entidades_criadas += len(boletim_ids)
    return stats


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bos-jsonl", required=True, type=Path)
    parser.add_argument("--ner-jsonl", required=True, type=Path)
    parser.add_argument("--pipeline-run-id", required=True)
    parser.add_argument("--stage", default="production", choices=["staging", "production", "archived"])
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--limit", type=int, default=None, help="Processa só os N primeiros registros (teste)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-file", type=Path, default=Path("import_bo_ner.log"))
    parser.add_argument("--summary-json", type=Path, default=None, help="summary.json do run (opcional, enriquece o ModelRegistry)")
    args = parser.parse_args()

    setup_logging(args.log_file)
    logger.info("Carregando bos_reconstruidos de %s...", args.bos_jsonl)
    bos_index = load_bos_index(args.bos_jsonl)
    logger.info("  %d BOs indexados por bo_id", len(bos_index))

    run_summary = {}
    summary_path = args.summary_json or (args.ner_jsonl.parent / "summary.json")
    if summary_path.exists():
        run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        logger.info("summary.json carregado de %s", summary_path)

    generated_at = datetime.fromtimestamp(args.ner_jsonl.stat().st_mtime, tz=timezone.utc)
    logger.info("generated_at (mtime de %s): %s", args.ner_jsonl.name, generated_at.isoformat())

    model_id = await get_or_create_model_registry(args.stage, run_summary, args.dry_run)

    total = count_lines(args.ner_jsonl)
    if args.limit:
        total = min(total, args.limit)
    logger.info("Processando %d registros de %s em lotes de %d", total, args.ner_jsonl.name, args.batch_size)

    unidade_cache: dict[str, int] = {}
    overall = Stats()
    start = time.monotonic()

    batch: list[dict] = []
    processed = 0

    with open(args.ner_jsonl, encoding="utf-8") as f, tqdm(total=total, unit="BO", desc="Importando") as pbar:
        async def run_batch(current_batch: list[dict]) -> None:
            nonlocal overall
            try:
                batch_stats = await process_batch(
                    current_batch, bos_index, model_id, args.pipeline_run_id, generated_at, unidade_cache, args.dry_run
                )
                overall.boletins_criados += batch_stats.boletins_criados
                overall.boletins_existentes += batch_stats.boletins_existentes
                overall.entidades_criadas += batch_stats.entidades_criadas
                overall.entidades_existentes += batch_stats.entidades_existentes
                overall.unidades_criadas += batch_stats.unidades_criadas
                overall.doc_id_sem_bo += batch_stats.doc_id_sem_bo
            except Exception:
                doc_ids = [r["doc_id"] for r in current_batch]
                logger.exception("Lote falhou (doc_ids=%s) — pulado, seguro re-rodar depois", doc_ids[:5])
                overall.lotes_falhos += 1
                overall.registros_em_lotes_falhos += len(current_batch)
                overall.doc_ids_falhos.extend(doc_ids)
            pbar.update(len(current_batch))
            pbar.set_postfix(
                criados=overall.boletins_criados,
                existentes=overall.boletins_existentes,
                falhas=overall.lotes_falhos,
            )

        for line in f:
            line = line.strip()
            if not line:
                continue
            batch.append(json.loads(line))
            processed += 1

            if len(batch) >= args.batch_size:
                await run_batch(batch)
                batch = []

            if args.limit and processed >= args.limit:
                break

        if batch:
            await run_batch(batch)

    elapsed = time.monotonic() - start
    overall.log_summary(elapsed, total)


if __name__ == "__main__":
    asyncio.run(main())
