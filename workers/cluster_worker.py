"""Consumes cluster:tasks (mensagem-gatilho, payload ignorado) e roda o job
completo de clustering — porta clusters_por_natureza de
Maynara_cluster_v0/motor_linkage.py de "consulta pontual filtrada por uma
natureza" pra "job em lote sobre o grafo inteiro de gold.bo_relacionado".

DIVERGÊNCIA DELIBERADA do motor original: clusters_por_natureza() sempre
filtra por UMA natureza antes de montar o grafo — um cluster nunca mistura
furto com roubo. Aqui o grafo é construído sobre TODAS as arestas
score >= cluster_score_threshold de uma vez, sem esse filtro — decisão
explícita, não uma omissão (um BO pode legitimamente conectar através de
naturezas diferentes, ex. mesmo documento aparecendo num furto e depois
numa ameaça).

Threshold: 0.9, o valor que o README do Maynara_cluster_v0 chama de
"calibrado nesta versão" — não o default 0.5 embutido no código-fonte de
clusters_por_natureza() (os dois divergem no pacote original).

Não incremental por natureza: precisa do grafo inteiro pra componentes
conexos ficarem corretos, não dá pra processar por mensagem-por-par sem
reprocessar tudo mesmo — daí o modelo de disparo por gatilho único.

Achado conhecido, NÃO mascarado aqui: o mesmo ruído de NER que inflou os
scores de linkage (ver linkage_scoring_worker.py — OBJETO_VEICULO/
DOCUMENTO capturando categoria genérica) colapsa boa parte do grafo num
componente gigante via conexões transitivas (A-B via "Motocicleta", B-C via
"Rg", A e C acabam no mesmo cluster sem relação direta nenhuma). Confirmado
com dado real: 490.055 arestas -> 153 componentes, um deles com 4.634 dos
5.013 nós. O worker grava esse resultado como está — o código está
correto, o problema é upstream (qualidade do NER), não algo pra filtrar ou
mascarar na camada de clustering.

gold.clusters.is_current opera no nível do CLUSTER INTEIRO (ver
app/modules/gold/models.py docstring): toda vez que este job roda, TODOS os
clusters da leva anterior são marcados is_current=false em lote antes da
leva nova entrar — não há chave lógica por cluster individual pra
supersede-then-insert par a par como bo_embedding/bo_relacionado.
"""
import asyncio

from networkx.utils.union_find import UnionFind
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from services.gold import get_or_create_model_id
from workers.base_worker import BaseWorker

MODEL_NAME = "bo_linkage_cluster"
MODEL_VERSION = "v1_componentes_conexos"

# Advisory-lock key for the whole-cluster supersede below ("bo_clust" as hex,
# arbitrary but fixed). Any other advisory lock in this database must not
# reuse it.
CLUSTER_SUPERSEDE_LOCK_KEY = 0x626F5F636C757374


class ClusterWorker(BaseWorker):
    def __init__(self, cfg=settings):
        super().__init__(cfg.cluster_stream, cfg)

    async def _carregar_arestas(self, db: AsyncSession) -> list[tuple[int, int]]:
        # shortcut: fetchall de todas as arestas (~60 MB em 490k, ~600 MB em
        # 5M). Trocar por db.stream() + union-find incremental quando o grafo
        # passar de ~1M arestas.
        result = await db.execute(
            text(
                """
                SELECT id_bo_origem, id_bo_destino
                FROM gold.bo_relacionado
                WHERE is_current AND score_sim >= :threshold
                """
            ),
            {"threshold": self.cfg.cluster_score_threshold},
        )
        return [(row[0], row[1]) for row in result.fetchall()]

    def _montar_componentes(self, arestas: list[tuple[int, int]]) -> list[list[int]]:
        # Union-find, não nx.Graph: o resultado é o mesmo conjunto de
        # componentes, mas não há dict de adjacência para 490k arestas
        # (centenas de MB, GBs a 5M) — só um nó por BO.
        grupos = UnionFind()
        for origem, destino in arestas:
            grupos.union(origem, destino)
        componentes = [sorted(c) for c in grupos.to_sets()]
        componentes.sort(key=len, reverse=True)
        return componentes

    async def process(self, data: dict) -> None:
        """Payload da mensagem é ignorado — qualquer mensagem em
        cluster:tasks dispara o job completo. Ver docstring do módulo."""
        async with self.db_session_factory() as db:  # type: AsyncSession
            arestas = await self._carregar_arestas(db)

        if not arestas:
            self.logger.info("Nenhuma aresta com score >= %.2f — nada a clusterizar.", self.cfg.cluster_score_threshold)
            return

        componentes = await asyncio.to_thread(self._montar_componentes, arestas)
        self.logger.info(
            "Grafo: %d arestas -> %d componentes conexos (maior: %d membros)",
            len(arestas), len(componentes), len(componentes[0]) if componentes else 0,
        )

        async with self.db_session_factory() as db:  # type: AsyncSession
            async with db.begin():
                # Two cluster jobs running at once would interleave their
                # batch supersede + insert: whichever commits last decides
                # which generation is is_current, so the older run can end up
                # current (and a reader can see two). The transaction-scoped
                # lock serializes them and releases on commit/rollback;
                # is_current has no per-row key to supersede on, unlike
                # bo_embedding/bo_relacionado.
                await db.execute(
                    text("SELECT pg_advisory_xact_lock(:key)"),
                    {"key": CLUSTER_SUPERSEDE_LOCK_KEY},
                )

                model_id = await get_or_create_model_id(
                    db,
                    MODEL_NAME,
                    MODEL_VERSION,
                    {
                        "algoritmo": "union-find",
                        "score_threshold": self.cfg.cluster_score_threshold,
                    },
                )

                # is_current no nível do CLUSTER INTEIRO — supersede em
                # lote da leva anterior, não par a par (ver docstring do
                # módulo e de app/modules/gold/models.py).
                await db.execute(
                    text(
                        "UPDATE gold.clusters SET is_current = false, superseded_at = now() WHERE is_current = true"
                    )
                )

                pipeline_run_id = data.get("pipeline_run_id")
                total_membros = 0
                for membros in componentes:
                    result = await db.execute(
                        text(
                            """
                            INSERT INTO gold.clusters (model_id, generated_at, pipeline_run_id)
                            VALUES (:model_id, now(), :pipeline_run_id)
                            RETURNING id_cluster
                            """
                        ),
                        {"model_id": model_id, "pipeline_run_id": pipeline_run_id},
                    )
                    id_cluster = result.scalar_one()

                    # executemany: um round trip por cluster em vez de um por
                    # membro (5.013 hoje), na mesma transação que segura o
                    # UPDATE de supersede.
                    await db.execute(
                        text(
                            """
                            INSERT INTO gold.bo_cluster (cluster_id, boletim_id)
                            VALUES (:cluster_id, :boletim_id)
                            """
                        ),
                        [
                            {"cluster_id": id_cluster, "boletim_id": boletim_id}
                            for boletim_id in membros
                        ],
                    )
                    total_membros += len(membros)

        self.logger.info(
            "Clustering gravado: %d clusters, %d BOs agrupados", len(componentes), total_membros
        )

    def run(self) -> None:
        try:
            super().run()
        finally:
            self.logger.info("Cluster worker finished")
