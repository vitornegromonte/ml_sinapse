#!/usr/bin/env python3
"""Dispara o ClusterWorker — publica uma mensagem-gatilho em cluster:tasks.

O payload é ignorado pelo worker (ver workers/cluster_worker.py):
ao receber QUALQUER mensagem, ele busca todas as arestas correntes de
gold.bo_relacionado com score >= cluster_score_threshold, monta o grafo
completo, extrai componentes conexos, e grava como uma leva nova em
gold.clusters/gold.bo_cluster (supersedindo a leva anterior em lote).

Rodar de novo é seguro a qualquer momento — sempre reflete o estado atual
de gold.bo_relacionado, não acumula estado entre execuções.

Uso:
    python scripts/trigger_clustering.py [--pipeline-run-id ID]

Requer:
    REDIS_URL — para publicar em cluster:tasks
"""
import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings  # noqa: E402
from services.task_publisher import enqueue  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger("trigger_clustering")


async def trigger(pipeline_run_id: str | None) -> None:
    run_id = pipeline_run_id or f"cluster_run_{int(time.time())}"
    msg_id = await enqueue(settings.cluster_stream, {"pipeline_run_id": run_id})
    logger.info("Gatilho publicado em %s [%s] (pipeline_run_id=%s)", settings.cluster_stream, msg_id, run_id)
    logger.info("Acompanhe o log do cluster-worker para o resultado (número de clusters/membros).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-run-id", default=None)
    args = parser.parse_args()

    asyncio.run(trigger(args.pipeline_run_id))


if __name__ == "__main__":
    main()
