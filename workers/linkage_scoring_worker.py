"""Consumes linkage:tasks (pares candidatos de BO, gerados por
scripts/backfill_linkage_scores.py) e grava o score de conexão em
gold.bo_relacionado.

Porta a matemática de Maynara_cluster_v0/motor_linkage.py (_componentes/
_feats/score_par) do índice em memória do notebook original para um worker
sem estado, montando o "perfil" de cada BO sob demanda a partir do que já
está em Postgres (gold.bo_embedding, gold.bo_entidades, silver.boletim) em
vez de manter tudo carregado num objeto Python.

SCHEMA_ENTIDADES abaixo (e ENTIDADES_ID/regras de normalização, agora em
services/entity_normalization.py) têm que ficar idênticos ao motor original —
definem o espaço de features que o classificador
(artifacts/bo_linkage_classifier_v1.joblib) foi treinado sobre. Reordenar ou
renomear um tipo muda o layout do vetor de perfil_tipos() e invalida
silenciosamente todo score que o classificador produzir.

sim_geo: sempre 0.0 hoje — nenhum pipeline de geocoding popula
silver.boletim.geom ainda (ver Task.md). Mesmo comportamento "bloco
neutro" que o motor original já tem para um BO sem coordenada, só que hoje
vale pra todos, não só alguns.

gold.bo_relacionado tem CHECK(id_bo_origem < id_bo_destino) — diferente do
motor original (que usa frozenset({id_a, id_b}), sem ordem imposta), o
schema exige o par normalizado antes do INSERT.
"""
import asyncio
from collections import OrderedDict

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from models.embedding_model import SentenceTransformerEncoder
from models.tasks import LinkageTask
from services.entity_normalization import (
    ENTIDADES_ID,
    normalizar_valor_entidade,
)
from services.gold import get_or_create_model_id
from workers.base_worker import BaseWorker

MODEL_NAME = "bo_linkage_classifier"
MODEL_VERSION = "v1"

# ------------------------------------------------------------
# SCHEMA DE ENTIDADES — idêntico a motor_linkage.py, não reordenar/renomear.
# ------------------------------------------------------------
SCHEMA_ENTIDADES = [
    "PESSOA_NOME", "PESSOA_DESCRICAO", "PESSOA_MENCAO", "ORGAO", "CARGO",
    "DOCUMENTO", "OBJETO_DOCUMENTO", "PLACA", "BO_ID", "DATA", "HORA",
    "VALOR_MONETARIO", "VALOR_MEDIDA", "VALOR_TEMPORAL",
    "EMAIL", "TELEFONE", "INSTAGRAM", "URL",
    "LOCAL_AREA", "LOCAL_ESTABELECIMENTO", "LOCAL_COMODO",
    "OBJETO_GERAL", "OBJETO_VEICULO", "OBJETO_ARMA",
    "ENTORPECENTE", "OBJETO_ENTORPECENTE", "OCORRENCIA", "ARTIGO_LEI",
    "CONTA_BANCARIA", "AGENCIA_BANCARIA", "DOCUMENTO_BOLETO",
    "CHAVE_PIX", "TRANSACAO_FINANCEIRA",
    "BO_COMPLEMENTADO",
]
N_ENT_TIPOS = len(SCHEMA_ENTIDADES)
ENT2IDX = {e: i for i, e in enumerate(SCHEMA_ENTIDADES)}

TIPOS_LOCAL_EXATO = {"LOCAL_AREA", "LOCAL_ESTABELECIMENTO", "LOCAL_COMODO"}
TIPOS_DESCRITIVOS_PADRAO = {"PESSOA_DESCRICAO", "OBJETO_GERAL", "OBJETO_ARMA", "OBJETO_VEICULO"}

def _perfil_tipos_vetor(entidades_por_tipo: dict) -> np.ndarray:
    v = np.zeros(N_ENT_TIPOS)
    for tipo in entidades_por_tipo:
        if tipo in ENT2IDX:
            v[ENT2IDX[tipo]] = 1.0
    norma = np.linalg.norm(v)
    return v / norma if norma > 0 else v


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine sem o overhead do sklearn: é uma chamada por par (490k no
    backfill). Não dá pra trocar por np.dot puro — encode() do
    sentence-transformers NÃO normaliza por padrão, então assumir norma 1
    mudaria o score em silêncio."""
    norma_a, norma_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if not norma_a or not norma_b:
        return 0.0
    return float(np.dot(a, b)) / (norma_a * norma_b)


def _cos_matriz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine entre cada linha de `a` e cada linha de `b` — as descrições de um
    BO contra as do outro. Mesmo motivo do _cos (uma chamada por par) e mesma
    semântica: linha de norma zero vira 0.0, como no sklearn."""
    sims = a @ b.T
    normas = np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1)[None, :]
    return np.divide(sims, normas, out=np.zeros_like(sims), where=normas > 0)


class LinkageScoringWorker(BaseWorker):
    def __init__(self, cfg=settings):
        super().__init__(cfg.linkage_stream, cfg)
        self._encoder: SentenceTransformerEncoder | None = None
        self._clf = None
        # LRU por boletim_id — o worker é um processo só, então um dict basta.
        self._perfis: OrderedDict[int, dict] = OrderedDict()

    def _load_model(self) -> None:
        import joblib

        device = (
            f"cuda:{self.cfg.linkage_gpu_device}"
            if self._cuda_available()
            else "cpu"
        )
        self._encoder = SentenceTransformerEncoder(self.cfg.embedding_model_name, device=device)
        self._encoder.load()

        self.logger.info("Loading classifier from %s", self.cfg.linkage_classifier_path)
        self._clf = joblib.load(self.cfg.linkage_classifier_path)
        self.logger.info("Classifier loaded, n_features_in_=%s", getattr(self._clf, "n_features_in_", "?"))

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    async def _load_perfil(self, db: AsyncSession, boletim_id: int) -> dict | None:
        """Monta o que score_par precisa de um BO a partir do que já está
        persistido — equivalente ao que self.emb_texto[i]/self.entidades_por_bo
        etc. dão no motor original, só que buscado sob demanda.

        Só SQL: quem chama encoda as descrições depois de fechar a sessão."""
        result = await db.execute(
            text(
                """
                SELECT e.embedding::text, b.natureza, b.horario_ocorrencia
                FROM gold.bo_embedding e
                JOIN silver.boletim b ON b.id_infopol = e.boletim_id
                WHERE e.boletim_id = :boletim_id AND e.is_current = true
                """
            ),
            {"boletim_id": boletim_id},
        )
        row = result.first()
        if row is None:
            return None
        embedding_raw, natureza, horario_ocorrencia = row
        # pgvector volta como texto literal ("[0.1,0.2,...]") via este
        # driver — não há conversão automática pra array sem registrar
        # pgvector.asyncpg.register_vector na conexão, o que exigiria acesso
        # à conexão asyncpg crua por trás da AsyncSession. Parse manual é
        # mais simples e não depende de driver interno.
        embedding = np.array(
            [float(x) for x in embedding_raw.strip("[]").split(",")], dtype=np.float32
        )

        result = await db.execute(
            text(
                "SELECT entidades FROM gold.bo_entidades WHERE boletim_id = :boletim_id AND is_current = true"
            ),
            {"boletim_id": boletim_id},
        )
        ent_row = result.first()
        entidades_raw = ent_row[0] if ent_row else []

        entidades_por_tipo: dict[str, set[str]] = {}
        descricoes: list[tuple[str, str]] = []
        for ent in entidades_raw:
            tipo = ent.get("label", "").upper()
            if tipo not in ENT2IDX:
                continue
            texto_original = ent.get("text", "")
            valor = normalizar_valor_entidade(texto_original, tipo)
            if valor:
                entidades_por_tipo.setdefault(tipo, set()).add(valor)
            if tipo in TIPOS_DESCRITIVOS_PADRAO:
                descricoes.append((tipo, texto_original))

        return {
            "boletim_id": boletim_id,
            "embedding": embedding,
            "entidades_por_tipo": entidades_por_tipo,
            "descricoes": descricoes,
            "emb_descricoes": None,
            "natureza": natureza,
            "data_fato": horario_ocorrencia,
        }

    async def _perfil(self, boletim_id: int) -> dict | None:
        """Perfil de um BO, cacheado: 490k pares reusam ~50k BOs e o encode das
        descrições é o custo de GPU dominante do pipeline. O encode roda com a
        sessão já fechada — não prende conexão do pool durante trabalho de GPU."""
        perfil = self._perfis.get(boletim_id)
        if perfil is not None:
            self._perfis.move_to_end(boletim_id)
            return perfil

        async with self.db_session_factory() as db:  # type: AsyncSession
            perfil = await self._load_perfil(db, boletim_id)
        if perfil is None:
            return None

        if perfil["descricoes"]:
            textos = [t for _, t in perfil["descricoes"]]
            perfil["emb_descricoes"] = await asyncio.to_thread(
                self._encoder.encode, textos
            )

        self._perfis[boletim_id] = perfil
        if len(self._perfis) > self.cfg.linkage_profile_cache_max:
            self._perfis.popitem(last=False)
        return perfil

    def _overlap_por_tipo(self, a: dict, b: dict, tipos: set) -> dict:
        overlaps = {}
        for tipo in tipos:
            vi = a["entidades_por_tipo"].get(tipo)
            vj = b["entidades_por_tipo"].get(tipo)
            if vi and vj:
                overlaps[tipo] = 1.0 if (vi & vj) else 0.0
        return overlaps

    def _sim_descricao(self, a: dict, b: dict) -> tuple:
        emb_a, emb_b = a["emb_descricoes"], b["emb_descricoes"]
        if emb_a is None or emb_b is None or len(emb_a) == 0 or len(emb_b) == 0:
            return 0.0, None
        sims = _cos_matriz(emb_a, emb_b)
        pos = np.unravel_index(np.argmax(sims), sims.shape)
        melhor = float(sims[pos])
        par_textos = (a["descricoes"][pos[0]][1], b["descricoes"][pos[1]][1])
        return melhor, par_textos

    def _componentes(self, a: dict, b: dict) -> dict:
        sim_texto = _cos(a["embedding"], b["embedding"])

        # sim_geo sempre 0.0 — ver docstring do módulo (sem geocoding ainda).
        sim_geo_bruta = 0.0

        if a["data_fato"] is not None and b["data_fato"] is not None:
            d_dias = abs((a["data_fato"] - b["data_fato"]).days)
            sim_data_bruta = float(np.exp(-d_dias / self.cfg.linkage_tau_dias))
        else:
            sim_data_bruta = 0.0

        overlaps_id = self._overlap_por_tipo(a, b, ENTIDADES_ID)
        sim_identificador = max(overlaps_id.values(), default=0.0)

        overlaps_local = self._overlap_por_tipo(a, b, TIPOS_LOCAL_EXATO)
        sim_local_exato = max(overlaps_local.values(), default=0.0)

        sim_descricao, par_descricao = self._sim_descricao(a, b)

        pi, pj = _perfil_tipos_vetor(a["entidades_por_tipo"]), _perfil_tipos_vetor(b["entidades_por_tipo"])
        sim_perfil = float(np.dot(pi, pj))

        motivos = [t for t, v in {**overlaps_id, **overlaps_local}.items() if v == 1.0]
        if par_descricao and sim_descricao >= self.cfg.linkage_limiar_descricao:
            motivos.append(f"descrição parecida ('{par_descricao[0]}' ~ '{par_descricao[1]}')")

        descricao_conta = sim_descricao if sim_descricao >= self.cfg.linkage_limiar_descricao else 0.0
        piso_relevancia = max(sim_texto, sim_identificador, sim_local_exato, descricao_conta)
        sim_geo = sim_geo_bruta * piso_relevancia
        sim_data = sim_data_bruta * piso_relevancia

        return {
            "sim_texto": sim_texto, "sim_geo": sim_geo, "sim_data": sim_data,
            "sim_identificador": sim_identificador, "sim_local_exato": sim_local_exato,
            "sim_descricao": sim_descricao, "sim_perfil": sim_perfil, "motivos": motivos,
        }

    def _feats(self, c: dict) -> list:
        # Ordem EXATA esperada pelo classificador — não reordenar.
        return [
            c["sim_texto"], c["sim_geo"], c["sim_data"], c["sim_identificador"],
            c["sim_local_exato"], c["sim_descricao"], c["sim_perfil"],
        ]

    async def process(self, data: dict) -> None:
        if self._encoder is None or self._clf is None:
            await asyncio.to_thread(self._load_model)

        task = LinkageTask(**data)
        id_a, id_b = task.id_bo_origem, task.id_bo_destino
        if id_a == id_b:
            self.logger.warning("Skipping self-pair boletim_id=%s", id_a)
            return
        # CHECK(id_bo_origem < id_bo_destino) — normaliza antes de qualquer coisa.
        if id_a > id_b:
            id_a, id_b = id_b, id_a

        perfil_a = await self._perfil(id_a)
        perfil_b = await self._perfil(id_b)

        if perfil_a is None or perfil_b is None:
            self.logger.warning(
                "Skipping pair (%s, %s) — embedding not ready for one or both sides", id_a, id_b
            )
            return

        componentes = await asyncio.to_thread(self._componentes, perfil_a, perfil_b)
        feats = self._feats(componentes)
        score = float(self._clf.predict_proba([feats])[0][1])

        metricas = {
            "sim_texto": round(componentes["sim_texto"], 4),
            "sim_geo": round(componentes["sim_geo"], 4),
            "sim_data": round(componentes["sim_data"], 4),
            "sim_identificador": round(componentes["sim_identificador"], 4),
            "sim_local_exato": round(componentes["sim_local_exato"], 4),
            "sim_descricao": round(componentes["sim_descricao"], 4),
            "sim_perfil": round(componentes["sim_perfil"], 4),
            "motivos": componentes["motivos"],
        }

        import json

        async with self.db_session_factory() as db:  # type: AsyncSession
            async with db.begin():
                model_id = await get_or_create_model_id(
                    db,
                    MODEL_NAME,
                    MODEL_VERSION,
                    {
                        "algoritmo": "LogisticRegression",
                        "C": 1000.0,
                        "class_weight": "balanced",
                    },
                )

                await db.execute(
                    text(
                        """
                        UPDATE gold.bo_relacionado
                        SET is_current = false, superseded_at = now()
                        WHERE id_bo_origem = :id_a AND id_bo_destino = :id_b AND is_current = true
                        """
                    ),
                    {"id_a": id_a, "id_b": id_b},
                )
                await db.execute(
                    text(
                        """
                        INSERT INTO gold.bo_relacionado
                            (id_bo_origem, id_bo_destino, score_sim, metricas, model_id, generated_at, pipeline_run_id)
                        VALUES
                            (:id_a, :id_b, :score, CAST(:metricas AS jsonb), :model_id, now(), :pipeline_run_id)
                        """
                    ),
                    {
                        "id_a": id_a,
                        "id_b": id_b,
                        "score": score,
                        "metricas": json.dumps(metricas),
                        "model_id": model_id,
                        "pipeline_run_id": task.pipeline_run_id,
                    },
                )

        self.logger.info(
            "Score written: (%s, %s) score=%.3f motivos=%s", id_a, id_b, score, componentes["motivos"]
        )

    def run(self) -> None:
        try:
            super().run()
        finally:
            if self._encoder is not None:
                self._encoder.unload()
            self.logger.info("Linkage scoring worker finished")
