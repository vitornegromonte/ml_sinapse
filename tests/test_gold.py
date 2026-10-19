"""get_or_create_model_id — the rollback-safety and race behaviour can't be
exercised without Postgres, so this pins the statement sequence with a stub
session (see services/gold.py for why the per-worker cache was removed)."""
from unittest.mock import AsyncMock, MagicMock

from services.gold import get_or_create_model_id


def _result(first=None, scalar=None):
    m = MagicMock()
    m.first.return_value = first
    m.scalar_one_or_none.return_value = scalar
    return m


async def test_returns_existing_id_without_inserting():
    db = MagicMock()
    db.execute = AsyncMock(side_effect=[_result(first=(7,))])

    assert await get_or_create_model_id(db, "m", "v1", {"a": 1}) == 7
    assert db.execute.await_count == 1


async def test_inserts_when_absent():
    db = MagicMock()
    db.execute = AsyncMock(side_effect=[_result(first=None), _result(scalar=11)])

    assert await get_or_create_model_id(db, "m", "v1") == 11
    assert db.execute.await_count == 2
    assert "ON CONFLICT DO NOTHING" in db.execute.call_args_list[1].args[0]


async def test_reselects_after_losing_the_race():
    db = MagicMock()
    db.execute = AsyncMock(
        side_effect=[_result(first=None), _result(scalar=None), _result(first=(22,))]
    )

    assert await get_or_create_model_id(db, "m", "v1") == 22
    assert db.execute.await_count == 3
