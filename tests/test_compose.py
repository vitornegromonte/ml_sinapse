"""docker-compose.yml: every worker exposes metrics and has the shared healthcheck.

A worker added later without them would run unmonitored, so this fails the suite.
"""
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

COMPOSE = Path(__file__).resolve().parents[1] / "docker-compose.yml"
WORKERS = ("ocr", "llm", "ner", "embedding", "linkage", "cluster", "scoring")


@pytest.fixture(scope="module")
def services():
    return yaml.safe_load(COMPOSE.read_text())["services"]


def test_every_worker_has_a_compose_service(services):
    """Arrange: the compose file and the worker names in run_worker.py.
    Act: look for a service that runs each worker.
    Assert: every worker has one.
    """
    commands = {" ".join(s["command"].split()[-1:]) for s in services.values()}

    assert set(WORKERS) <= commands


@pytest.mark.parametrize("worker", WORKERS)
def test_a_worker_service_sets_the_metrics_port(services, worker):
    """Arrange: the compose service of a worker.
    Act: read its environment.
    Assert: METRICS_PORT=9100 is set, since the healthcheck probes that port.
    """
    service = services[f"{worker}-worker"]

    assert "METRICS_PORT=9100" in service["environment"]


@pytest.mark.parametrize("worker", WORKERS)
def test_a_worker_service_has_the_shared_healthcheck(services, worker):
    """Arrange: the compose service of a worker.
    Act: read its healthcheck.
    Assert: it probes /healthz on port 9100 and allows time for the first model load.
    """
    check = services[f"{worker}-worker"]["healthcheck"]

    assert "localhost:9100/healthz" in " ".join(check["test"])
    assert check["test"][0] == "CMD"
    assert int(check["start_period"].rstrip("s")) >= 300


def test_the_metrics_port_is_not_published_to_the_host(services):
    """Arrange: every service in the compose file.
    Act: look for a published port.
    Assert: none, so the endpoint stays on the container network.
    """
    for name, service in services.items():
        assert "ports" not in service, f"{name} publishes ports"
