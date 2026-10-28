"""run_worker must resolve a worker module only for the name it was asked for."""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_importing_run_worker_loads_no_worker_module():
    """Arrange: a fresh interpreter that imports run_worker and lists workers.*
    Act: run the import in a subprocess.
    Assert: it exits cleanly and imports no worker module — starting the OCR
    container must not pay for torch/spaCy/sklearn from the other five.
    """
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, run_worker; "
            "print(' '.join(m for m in sys.modules if m.startswith('workers')))",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO)},
    )

    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""


def test_resolve_returns_the_class_by_name():
    """Arrange: the worker registry imported from run_worker.
    Act: resolve the "cluster" worker by name.
    Assert: the registry maps it to ClusterWorker and resolve returns that class.
    """
    import run_worker

    assert run_worker.WORKERS["cluster"] == "workers.cluster_worker:ClusterWorker"
    assert run_worker.resolve("cluster").__name__ == "ClusterWorker"
