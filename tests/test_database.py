"""create_db_engine must recycle connections the server closed while a worker
was idle, and must not hang forever dialing an unreachable database."""
from unittest.mock import patch

from config import Settings
from services.database import create_db_engine


def test_engine_enables_pre_ping_and_a_bounded_connect_timeout():
    cfg = Settings(ml_database_url="postgresql+asyncpg://u:p@db:5432/d")

    with patch("services.database.create_async_engine") as create:
        create_db_engine(cfg)

    url, kwargs = create.call_args[0][0], create.call_args[1]
    assert url == "postgresql+asyncpg://u:p@db:5432/d"
    assert kwargs["echo"] is False
    assert kwargs["pool_pre_ping"] is True
    assert kwargs["connect_args"] == {"timeout": 30}
