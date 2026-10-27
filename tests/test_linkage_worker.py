"""Per-BO profile cache in the linkage scorer: 490k candidate pairs reuse ~50k
BOs, and re-encoding a BO's descriptions is the pipeline's dominant GPU cost.
See TASK.md B13."""
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from workers.linkage_scoring_worker import LinkageScoringWorker

DIM = 8


class _Result:
    def __init__(self, row):
        self._row = row

    def first(self):
        return self._row


class _Ctx:
    def __init__(self, value):
        self._value = value

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    """Answers the two profile SELECTs from `perfis` and records everything
    else (the supersede + insert of the score)."""

    def __init__(self, perfis):
        self.perfis = perfis
        self.profile_queries = 0
        self.writes = []

    async def execute(self, stmt, params=None):
        sql = str(stmt)
        if "gold.bo_embedding" in sql:
            perfil = self.perfis.get(params["boletim_id"])
            self.profile_queries += 1
            return _Result(perfil[0] if perfil else None)
        if "gold.bo_entidades" in sql:
            perfil = self.perfis.get(params["boletim_id"])
            return _Result(perfil[1] if perfil else None)
        self.writes.append((sql, params))
        return _Result(None)

    def begin(self):
        return _Ctx(self)


def _perfil_rows(boletim_id, n_descricoes=1):
    vetor = "[" + ",".join(f"{x:.1f}" for x in np.eye(DIM)[boletim_id % DIM]) + "]"
    entidades = [
        {"label": "PESSOA_DESCRICAO", "text": f"descricao {boletim_id}-{k}"}
        for k in range(n_descricoes)
    ]
    return (vetor, "natureza", None), (entidades,)


@pytest.fixture
def session():
    return _FakeSession(
        {
            1: _perfil_rows(1),
            2: _perfil_rows(2, n_descricoes=2),
            3: _perfil_rows(3),
            4: _perfil_rows(4),
        }
    )


@pytest.fixture
def worker(session):
    with patch(
        "workers.linkage_scoring_worker.get_or_create_model_id",
        new=AsyncMock(return_value=7),
    ):
        w = LinkageScoringWorker()
        w._encoder = MagicMock()
        # Um vetor por descrição, para casar com a matriz que _sim_descricao indexa.
        w._encoder.encode.side_effect = lambda textos: np.ones((len(textos), DIM))
        w._clf = MagicMock()
        w._clf.predict_proba.return_value = np.array([[0.1, 0.9]])
        w.db_session_factory = lambda: _Ctx(session)
        yield w


async def test_encodes_each_bo_once_across_pairs(worker, session):
    """BO 1 is an endpoint of both pairs but must be loaded and encoded once."""
    await worker.process({"id_bo_origem": 1, "id_bo_destino": 2})
    await worker.process({"id_bo_origem": 1, "id_bo_destino": 3})

    assert worker._encoder.encode.call_count == 3  # BOs 1, 2, 3 — not 4
    assert session.profile_queries == 3
    assert len(session.writes) == 4  # supersede + insert, per pair


async def test_reversed_pair_reuses_the_cache(worker):
    """The CHECK(id_bo_origem < id_bo_destino) swap must not defeat the cache."""
    await worker.process({"id_bo_origem": 3, "id_bo_destino": 1})
    await worker.process({"id_bo_origem": 1, "id_bo_destino": 3})

    assert worker._encoder.encode.call_count == 2


async def test_cache_evicts_the_least_recently_used(worker, monkeypatch):
    """With a 2-entry cache, BO 2 is evicted by BO 3 and encoded a second time —
    eviction costs work, never a wrong score."""
    monkeypatch.setattr(worker.cfg, "linkage_profile_cache_max", 2)

    for origem, destino in [(1, 2), (1, 3), (1, 2)]:
        await worker.process({"id_bo_origem": origem, "id_bo_destino": destino})

    assert worker._encoder.encode.call_count == 4  # 1, 2, 3, then 2 again


async def test_written_pair_is_ordered_and_scored(worker, session):
    await worker.process({"id_bo_origem": 2, "id_bo_destino": 1})

    insert = session.writes[1][1]
    assert (insert["id_a"], insert["id_b"]) == (1, 2)
    assert insert["score"] == pytest.approx(0.9)


async def test_pair_without_embedding_is_skipped_and_not_cached(worker, session):
    """A BO whose embedding is not ready yet must not be cached as a miss —
    the next attempt has to re-query it."""
    await worker.process({"id_bo_origem": 1, "id_bo_destino": 99})
    await worker.process({"id_bo_origem": 1, "id_bo_destino": 99})

    assert session.writes == []
    assert session.profile_queries == 3  # BO 1 once, BO 99 twice
