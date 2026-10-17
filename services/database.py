"""Async Postgres connection for ML workers.

Workers connect as `sinapse_ml` (member of the `ml_worker` role: read
`silver`, write `gold` — never `silver`). See
backend/alembic/versions/0002_schema_roles_and_grants.py and
0004_add_ml_login_role.py for the role/grant definitions.

Deliberately standalone: this package does not import backend's ORM models or
app.core.database. The two services are deployed independently (separate
Dockerfile/requirements.txt) and shouldn't share a Python package boundary.
Per-worker Gold table models get defined here as each worker's DB writes get
built out (this module only provides the connection itself).
"""

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from config import Settings, settings


def create_db_engine(cfg: Settings = settings) -> AsyncEngine:
    return create_async_engine(cfg.ml_database_url, echo=False)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
