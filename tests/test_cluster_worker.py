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
