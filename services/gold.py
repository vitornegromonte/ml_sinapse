"""Shared Gold-table helpers for the ML workers.

Raw SQL via text(), same rationale as services/database.py: this package and
the backend are separate deployables that don't share a Python package
boundary.
"""
import json

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def get_or_create_model_id(
    db: AsyncSession,
    name: str,
    version: str,
    hyperparameters: dict | None = None,
) -> int:
    """Return the gold.model_registry id for (name, version), creating it if absent.

    Call this from inside the caller's transaction — it does not commit. It
    deliberately does NOT cache the result: an id read with RETURNING belongs
    to a row that is not committed yet, so caching it on the worker outlives a
    rollback and poisons every later message (the INSERT lands, RETURNING
    hands back an id, a later statement in the same transaction fails, the row
    disappears — the cached id does not). One SELECT per message is cheap next
    to the real work.

    Concurrency-safe: `ON CONFLICT DO NOTHING` (no target, so it doesn't
    assume which columns are unique) waits on a conflicting uncommitted row —
    commits means no row is returned and the re-SELECT finds it; rolls back
    means our INSERT wins.
    """
    params = {"name": name, "version": version}
    select_id = text(
        "SELECT id FROM gold.model_registry WHERE name = :name AND version = :version"
    )

    row = (await db.execute(select_id, params)).first()
    if row is not None:
        return row[0]

    result = await db.execute(
        text(
            """
            INSERT INTO gold.model_registry (name, version, stage, hyperparameters)
            VALUES (:name, :version, 'production', :hyperparameters)
            ON CONFLICT DO NOTHING
            RETURNING id
            """
        ),
        {**params, "hyperparameters": json.dumps(hyperparameters or {})},
    )
    inserted = result.scalar_one_or_none()
    if inserted is not None:
        return inserted

    # Lost the race: another transaction committed the row between our SELECT
    # and our INSERT, so DO NOTHING suppressed ours.
    return (await db.execute(select_id, params)).first()[0]
