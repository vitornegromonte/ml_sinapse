"""The whole-cluster supersede must take its advisory lock as the first
statement of the write transaction. gold.clusters has no per-row key to
supersede on, so without the lock two concurrent jobs interleave (batch
supersede + insert) and the run that commits last decides which generation is
is_current — the older one can win, or a reader can see two."""
from unittest.mock import AsyncMock, MagicMock

from workers.cluster_worker import CLUSTER_SUPERSEDE_LOCK_KEY, ClusterWorker


class _FakeBegin:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, statements):
        self.statements = statements

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def begin(self):
        return _FakeBegin()

    async def execute(self, clause, params=None):
        self.statements.append((str(clause), params))
        result = MagicMock()
        result.first.return_value = (1,)
        result.scalar_one.return_value = 42
        return result


async def test_supersede_takes_advisory_lock_before_touching_clusters():
    """Arrange: a worker with a stub session recording every statement.
    Act: process one clustering trigger.
    Assert: pg_advisory_xact_lock with the shared key is the first statement,
    before the UPDATE that supersedes gold.clusters.
    """
    worker = ClusterWorker()
    statements = []
    worker.db_session_factory = lambda: _FakeSession(statements)
    worker._carregar_arestas = AsyncMock(return_value=[(1, 2)])
    worker._montar_componentes = MagicMock(return_value=[[1, 2]])

    await worker.process({})

    first_sql, first_params = statements[0]
    assert "pg_advisory_xact_lock" in first_sql
    assert first_params == {"key": CLUSTER_SUPERSEDE_LOCK_KEY}

    update_at = next(
        i for i, (sql, _) in enumerate(statements) if "UPDATE gold.clusters" in sql
    )
    assert update_at > 0  # lock first, batch supersede after


def test_montar_componentes_agrupa_sem_montar_grafo():
    """Arrange: edges with a shared node, a separate pair, and a duplicate.
    Act: build the connected components.
    Assert: two components, with duplicates collapsed and no graph object.
    """
    worker = ClusterWorker()

    assert worker._montar_componentes([(1, 2), (2, 3), (7, 8), (1, 2)]) == [[1, 2, 3], [7, 8]]


def test_montar_componentes_inclui_autolaco_e_ignora_no_sem_aresta():
    """Arrange: edges with a self-loop and no edge for node 4.
    Act: build the connected components.
    Assert: the self-loop node forms its own component and the isolated node is
    absent.
    """
    worker = ClusterWorker()

    # Mesmo resultado do nx.Graph anterior: 9 é nó (autolago), 4 não é.
    assert worker._montar_componentes([(1, 2), (9, 9)]) == [[1, 2], [9]]


async def test_membros_sao_inseridos_em_lote_por_cluster():
    """Arrange: two components, one with two members and one with one.
    Act: process one clustering trigger.
    Assert: one INSERT per cluster, each with that cluster's member rows.
    """
    worker = ClusterWorker()
    statements = []
    worker.db_session_factory = lambda: _FakeSession(statements)
    worker._carregar_arestas = AsyncMock(return_value=[(1, 2), (3, 4)])
    worker._montar_componentes = MagicMock(return_value=[[1, 2], [3]])

    await worker.process({})

    insercoes = [
        params for sql, params in statements if "INSERT INTO gold.bo_cluster" in sql
    ]
    assert insercoes == [
        [{"cluster_id": 42, "boletim_id": 1}, {"cluster_id": 42, "boletim_id": 2}],
        [{"cluster_id": 42, "boletim_id": 3}],
    ]
