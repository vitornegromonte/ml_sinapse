"""B15: o lote inteiro paga um encode() e três round trips, não um por BO.

O caminho de 1 item agora delega a process_batch([data]) — o mesmo código —
para não haver duas versões da supersede-then-insert.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from workers.embedding_worker import EmbeddingWorker


class _Vec:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


class _FakeBegin:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def begin(self):
        return _FakeBegin()

    async def execute(self, clause, params=None):
        sql = str(clause)
        self._conn.statements.append((sql, params))
        if "FROM silver.boletim" in sql:
            narratives = self._conn.narratives
            return _FakeResult(
                [(bid, narratives[bid]) for bid in params["ids"] if bid in narratives]
            )
        return _FakeResult([])


class _Conn:
    def __init__(self, narratives):
        self.narratives = narratives
        self.statements = []


@pytest.fixture
def worker():
    w = EmbeddingWorker()
    w._encoder = MagicMock()
    w._encoder.encode.return_value = [_Vec([0.1, 0.2]), _Vec([0.3, 0.4])]
    return w


@pytest.fixture
def db(worker):
    conn = _Conn({1: "texto um", 2: "texto dois"})
    worker.db_session_factory = lambda: _FakeSession(conn)
    return conn


@pytest.fixture
def model_id():
    with patch(
        "workers.embedding_worker.get_or_create_model_id",
        AsyncMock(return_value=7),
    ):
        yield


def _inserted(conn):
    return [p for s, p in conn.statements if "INSERT INTO gold.bo_embedding" in s]


async def test_batch_encodes_all_texts_in_one_call(worker, db, model_id):
    errors = await worker.process_batch([{"boletim_id": 1}, {"boletim_id": 2}])

    assert errors == [None, None]
    worker._encoder.encode.assert_called_once_with(["texto um", "texto dois"])
    updates = [p for s, p in db.statements if "UPDATE gold.bo_embedding" in s]
    assert updates == [{"ids": [1, 2]}]
    assert len(_inserted(db)) == 1
    assert [row["boletim_id"] for row in _inserted(db)[0]] == [1, 2]


async def test_batch_writes_a_repeated_bo_once(worker, db, model_id):
    """Duas mensagens do mesmo BO no mesmo lote não podem virar duas linhas
    current — o UPDATE único não supersede a linha que o próprio lote criou."""
    errors = await worker.process_batch([{"boletim_id": 1}, {"boletim_id": 1}])

    assert errors == [None, None]
    worker._encoder.encode.assert_called_once_with(["texto um"])
    assert len(_inserted(db)[0]) == 1


async def test_bo_without_narrative_is_skipped_not_encoded(worker, db, model_id):
    db.narratives = {1: "  "}

    errors = await worker.process_batch([{"boletim_id": 1}, {"boletim_id": 3}])

    assert errors == [None, None]
    worker._encoder.encode.assert_not_called()
    assert _inserted(db) == []


async def test_bad_payload_fails_only_its_item(worker, db, model_id):
    errors = await worker.process_batch([{"boletim_id": "abc"}, {"boletim_id": 1}])

    assert isinstance(errors[0], ValueError)
    assert errors[1] is None
    worker._encoder.encode.assert_called_once_with(["texto um"])


async def test_process_single_item_raises_the_batch_error(worker, db, model_id):
    with pytest.raises(ValueError):
        await worker.process({"boletim_id": "abc"})


async def test_encode_failure_is_reported_for_every_ready_item(worker, db, model_id):
    worker._encoder.encode.side_effect = RuntimeError("cuda oom")

    errors = await worker.process_batch([{"boletim_id": 1}, {"boletim_id": 2}])

    assert all(isinstance(e, RuntimeError) for e in errors)
    assert _inserted(db) == []
