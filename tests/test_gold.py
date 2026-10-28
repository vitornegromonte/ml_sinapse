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
    """Arrange: a stub session whose first SELECT finds model id 7.
    Act: get or create the model id.
    Assert: id 7 is returned and no insert is attempted.
    """
    db = MagicMock()
    db.execute = AsyncMock(side_effect=[_result(first=(7,))])

    assert await get_or_create_model_id(db, "m", "v1", {"a": 1}) == 7
    assert db.execute.await_count == 1


async def test_inserts_when_absent():
    """Arrange: a stub session where the SELECT misses and the insert returns 11.
    Act: get or create the model id.
    Assert: id 11 is returned, two statements run, and the insert uses
    ON CONFLICT DO NOTHING.
    """
    db = MagicMock()
    db.execute = AsyncMock(side_effect=[_result(first=None), _result(scalar=11)])

    assert await get_or_create_model_id(db, "m", "v1") == 11
    assert db.execute.await_count == 2
    # str(): execute() receives a sqlalchemy TextClause, not a str.
    assert "ON CONFLICT DO NOTHING" in str(db.execute.call_args_list[1].args[0])


async def test_reselects_after_losing_the_race():
    """Arrange: a stub session where the SELECT misses, the insert returns no
    row (another worker won the race), and the re-SELECT finds id 22.
    Act: get or create the model id.
    Assert: id 22 is returned after three statements.
    """
    db = MagicMock()
    db.execute = AsyncMock(
        side_effect=[_result(first=None), _result(scalar=None), _result(first=(22,))]
    )

    assert await get_or_create_model_id(db, "m", "v1") == 22
    assert db.execute.await_count == 3
