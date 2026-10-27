# ml_sinapse

Standalone ML service for SINAPSE — extracted from the `sinapse` monorepo
(`ml_service/` package + pipeline scripts + ADRs). Deploys independently of
the backend; the two sides talk through **Redis Streams** and shared
Postgres/MinIO, never a shared Python package.

## Workers

Each worker is its own process/container, consuming its own stream with the
`ml-workers` consumer group (retry → dead-letter → `XAUTOCLAIM` reclaim, see
`workers/base_worker.py`):

| Worker | Command | Consumes | Writes |
|---|---|---|---|
| `ocr` | `python run_worker.py ocr` | `ocr:tasks` | MinIO `bronze` + `ocr:tasks:results` |
| `llm` | `python run_worker.py llm` | `ocr:tasks:results` | MinIO `bronze` + `llm:tasks:results` |
| `ner` | `python run_worker.py ner` | `ner:tasks` | logs only (mock NER, ADR-005) |
| `embedding` | `python run_worker.py embedding` | `embedding:tasks` | `gold.bo_embedding` |
| `linkage` | `python run_worker.py linkage` | `linkage:tasks` | `gold.bo_relacionado` |
| `cluster` | `python run_worker.py cluster` | `cluster:tasks` (any msg = trigger) | `gold.clusters`, `gold.bo_cluster` |

## Contract with the backend (sinapse repo)

- **Postgres roles**: workers connect as `sinapse_ml` (`ml_worker` group) —
  reads `silver`, writes `gold`, **never writes silver**. Schema, grants and
  migrations live in `sinapse` (`backend/alembic/`, notably `0002` grants and
  `0004` ML login role). Silver writes triggered by ML output round-trip
  through the backend's own stream consumer.
- **Stream names**: defined in `config.py`, mirrored (by convention, not
  import) by `backend/app/core/queue.py`.
- **MinIO buckets**: `bronze` (PDFs in, OCR/LLM text out), `models`
  (weights synced to `/model-cache`, ADR-006), `mlflow-artifacts`.

## Run

Core stack (Postgres/Redis/MinIO/MLflow) stays in the `sinapse` repo — bring
it up first, then:

```bash
BACKEND_HOST=<backend machine IP> docker compose up --build
```

No default for `BACKEND_HOST` on purpose: compose fails fast instead of
building broken connection strings. Workers don't retry an unreachable
Postgres/Redis at startup. Extra env vars (all optional):
`POSTGRES_ML_WORKER_PASSWORD`, `MINIO_ROOT_USER`/`MINIO_ROOT_PASSWORD`,
`EMBEDDING_MODEL_NAME`, `LLM_MODEL_PATH`, `NER_MODEL_MINIO_PREFIX`.
Put them in a `.env` next to `docker-compose.yml` (never committed).

GPU note: `ocr`/`llm`/`embedding`/`linkage` reserve 1 GPU; `ner`/`cluster`
are CPU-only. `VLLM_WORKER_MULTIPROC_METHOD=spawn` is required where vLLM
runs (model load happens in a non-main thread).

## Develop / test

```bash
pip install -r requirements-dev.txt -r requirements-test.txt  # CPU-only: no GPU/CUDA/vLLM
python -m pytest -q          # mock-based, no GPU/DB/Redis needed
```

`requirements-test.txt` is the CPU-only subset used by CI
(`.github/workflows/tests.yml`); `requirements.txt` is only needed to actually
run a worker.

Pipeline drivers in `scripts/` (run from repo root):

| Script | Needs | Purpose |
|---|---|---|
| `backfill_embeddings.py` | `ML_DATABASE_URL`, `REDIS_URL` | enqueue BOs missing an embedding |
| `backfill_linkage_scores.py` | `ML_DATABASE_URL`, `REDIS_URL` | generate candidate pairs (top-K HNSW + strong-entity overlap) |
| `trigger_clustering.py` | `REDIS_URL` | fire connected-component clustering |
| `test_ocr_llm.py` | GPU | one-shot PDF → OCR → LLM smoke test, with the VRAM each stage added (`OCR_GPU_MEMORY_UTILIZATION` / `LLM_GPU_MEMORY_UTILIZATION` choose the vLLM budget) |
| `profile_models.py` | GPU | measure each worker's VRAM (`ocr`/`llm`/`embedding`/`linkage`) to decide the GPU routing — `--all --gpu 0`, one process per model |
| `import_bo_ner.py` | sibling `../sinapse` checkout + DB URLs | **legacy one-shot**: seeds `silver.boletim` + `gold.bo_entidades` through the backend ORM (already executed; kept as-is per decision) |
| `init_buckets.sh`, `download_test_images.py`, `publish_{ocr,ner}_task.py` | — | bucket bootstrap / fixture helpers |

## Docs

Design decisions: [`docs/adr-001-service-structure.md`](docs/adr-001-service-structure.md)
through `adr-010-bentoml-serving`. ADRs are historical records — paths in
them still say `ml_service/...` (this package was flattened to the repo
root) and `docs/backend/...` (those files stayed in the `sinapse` repo).

## Layout

Flattened from the monorepo: `workers/`, `models/`, `services/`,
`config.py`, `run_worker.py` at the root (imports are un-prefixed:
`from workers.base_worker import ...`). Compose builds from `.`.
