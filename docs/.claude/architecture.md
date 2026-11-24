# ML Sinapse — architecture

Standalone ML service for **SINAPSE**. It deploys separately from the
backend and shares no Python package with it. The two sides communicate only
through:

- **Redis Streams** — task queues and results
- **Postgres** — `silver` read-only for ML, `gold` writable for ML
- **MinIO** — source PDFs, ML artifacts, model weights

The backend and the core stack (Postgres/Redis/MinIO/MLflow) live in the
`sinapse` repo. This repo builds seven worker containers plus one-shot scripts.
The interactive companion is [`architecture.html`](../architecture.html)
(git-ignored, local only).

Decisions behind this design are in the [ADRs](./README.md).

---

## Deployment

`docker-compose.yml` starts one container per worker and connects back to the
backend machine through `BACKEND_HOST`. There are no local stateful services.

| Dependency | Endpoint used by this repo | Ownership |
|---|---|---|
| Redis Streams | `BACKEND_HOST:6380` | `sinapse` core stack |
| Postgres | `BACKEND_HOST:5433/sinapse_db` | `sinapse` core stack |
| MinIO | `BACKEND_HOST:9000` | `sinapse` core stack |
| MLflow | `http://mlflow:5000` (`config.py` default) | `sinapse` core stack |
| Backend API | not called (streams only) | `sinapse` repo |

The ML database role is `sinapse_ml` (member of `ml_worker`): **read
`silver`, write `gold`, never write `silver`**. Silver writes caused by ML
output round-trip through a backend-side stream consumer.

---

## Components

| Node | Code | Runtime | Consumes | Produces |
|---|---|---|---|---|
| SINAPSE backend | `sinapse` repo | FastAPI | result streams | `ocr:tasks`, `ner:tasks`, `scoring:tasks`, source PDFs in `bronze`; writes `silver` |
| Backfill / trigger | `scripts/` | one-shot CLI | `silver` + `gold` reads | `embedding:tasks`, `linkage:tasks`, `cluster:tasks` |
| Redis Streams | core stack | Redis | — | task/result streams, retry hashes, leases, `:dead` streams, OCR checkpoint hashes |
| MinIO | core stack | S3 | — | `bronze`, `models`, `mlflow-artifacts` |
| Postgres | core stack | PG + pgvector | — | `silver`, `gold` |
| OCR worker | `workers/ocr_worker.py` | GPU | `ocr:tasks` | `bronze/ml-results/ocr/…`, `ocr:tasks:results` |
| LLM worker | `workers/llm_worker.py` | GPU, batch | `ocr:tasks:results` | `bronze/ml-results/llm/…`, `llm:tasks` (see note) |
| NER worker | `workers/ner_worker.py` | CPU | `ner:tasks` | **logs only** (ADR-005 mock) |
| Embedding worker | `workers/embedding_worker.py` | GPU, batch | `embedding:tasks` | `gold.bo_embedding`, `gold.model_registry` |
| Linkage worker | `workers/linkage_scoring_worker.py` | GPU | `linkage:tasks` | `gold.bo_relacionado`, `gold.model_registry` |
| Cluster worker | `workers/cluster_worker.py` | CPU | `cluster:tasks` | `gold.clusters`, `gold.bo_cluster`, `gold.model_registry` |
| Scoring worker | `workers/scoring_worker.py` | CPU, Redis only | `scoring:tasks` | `scoring:tasks:results` |
| model-cache | Docker volume | — | — | `/model-cache` weights |
| MLflow | core stack | MLflow | — | optional model registry for NER and OCR (`NER_MODEL_URI`, `OCR_MODEL_URI`) |

The LLM worker publishes to the stream named by `llm_stream`, which is
`llm:tasks`. Older docs call it `llm:tasks:results`. The code is the source
of truth.

The tokenizer worker and the BentoML LLM server (ADR-001/003/010) were
removed. The embedding worker replaces tokenization. There is no synchronous
REST serving in this repo.

---

## Streams and failure semantics

Stream names come from `config.py`. The backend mirrors them by convention,
not by import.

| Stream | Producer | Consumer |
|---|---|---|
| `ocr:tasks` | backend | OCR worker |
| `ocr:tasks:results` | OCR worker | LLM worker, backend |
| `llm:tasks` | LLM worker | backend |
| `ner:tasks` | backend | NER worker |
| `embedding:tasks` | `backfill_embeddings.py`, backend | Embedding worker |
| `linkage:tasks` | `backfill_linkage_scores.py` | Linkage worker |
| `cluster:tasks` | `trigger_clustering.py` | Cluster worker |
| `scoring:tasks` | backend | Scoring worker |
| `scoring:tasks:results` | Scoring worker | backend |

Every worker inherits `workers/base_worker.py`:

- Consumer group `ml-workers`. Each worker makes one blocking `XREADGROUP` per stream.
- Workers with `supports_batch = True` (`embedding`, `llm`) read up to
  `batch_size` (8) messages and process them in one call. Other workers read
  one message at a time.
- A failed message stays **pending** (not acked). The retry counter is in the
  Redis hash `<stream>:retries`.
- `XAUTOCLAIM` reclaims a pending message after `claim_idle_ms` (30 s).
- While a message is in flight, the worker refreshes a lease key
  `<stream>:lease:<id>` every 10 s (TTL 30 s). A replica skips the reclaim
  when the lease is alive. This stops it from stealing slow work.
- A `PermanentTaskError` skips the retries. The worker copies the payload to
  `<stream>:dead` at once and acks it (ADR-011). Raise it only for tasks that
  can never succeed.
- After `max_retries` (3), the payload is copied to `<stream>:dead` and acked.
  `:dead` streams keep at most 10,000 entries and 30 days.
- The worker shuts down cleanly on `SIGTERM`/`SIGINT`.

Result streams (`ocr`, `llm`, `scoring`) use `MAXLEN ~50000`. This cap
protects RAM when the backend consumer is down. Messages over the cap are
dropped. The publisher logs a warning when the cap is reached.

Delivery is at-least-once, so every `process()` must be idempotent. Embedding
and linkage supersede the old row, then insert the new row, per logical key.
Clustering supersedes the whole previous batch.

---

## Flows

### 1. PDF ingest — OCR → LLM summary

1. Backend uploads the PDF to `bronze/bos/<year>/<id>/original.pdf`.
2. Backend publishes `{document_id, pdf_bucket, pdf_key}` to `ocr:tasks`.
3. The OCR worker receives the message through its consumer group.
4. OCR validates the payload. An invalid payload is a permanent failure and goes
   straight to DLQ. OCR then downloads the PDF from MinIO. A missing object goes
   to retry, then DLQ.
5. OCR splits pages with PyMuPDF. It keeps text blocks. It renders scanned
   pages and image blocks at `ocr_render_dpi` (default 200) and deskews them
   with OpenCV. More than `ocr_max_pdf_pages` (200) is a permanent failure.
   OCR sends the scanned pages to **Dharma-OCR-LITE** through vLLM in batches
   of `ocr_page_batch_size` (default 4; `OCR_PAGE_BATCH_SIZE=1` gives one page
   per model call). The PDF order is kept. While one batch is on the GPU, the
   worker renders and deskews the next batch. The model runs with
   `repetition_penalty` 1.05, the value in the model's own
   `generation_config.json`. Without it, some pages looped until the token cap.
   Dharma answers each page with JSON (`header`, `margin`, `footer`, `text`).
   The model wrapper keeps only `text` and returns it as plain text.
   `OCR_MODEL_NAME` selects another model, such as GLM-OCR, whose output is
   used as is. The OCR vLLM engine may use at most 15% of GPU memory
   (`ocr_gpu_memory_utilization`, default and maximum 0.15) and a context of
   `ocr_max_model_len` (default 24,576). At the model's native 128k context the
   engine does not start inside 15%.
   Each model call has a limit of `ocr_predict_timeout_seconds` (default 600).
   After a timeout the worker fails the message, stops, and exits the process.
   The container restart policy starts a new one (see ADR-011).
   After each finished batch, OCR saves the page texts in the Redis hash
   `<stream>:partial:<document_id>:<hash>` with a TTL of
   `ocr_partial_ttl_seconds` (default 86,400; 0 turns it off). The hash covers
   the PDF bytes, the model, the DPI, and the token cap. A retry of the same
   task reads that hash and does not render or OCR the saved pages again
   (ADR-012).
6. OCR writes the text to `bronze/ml-results/ocr/<document_id>`.
7. OCR publishes `{document_id, text, text_length}` to `ocr:tasks:results`.
   Then it deletes the checkpoint hash. A failure before this point keeps it.
8. The LLM worker consumes that stream.
9. The LLM builds the Portuguese summary prompt and generates with vLLM. The
   model is `maritaca-ai/S5-nano` (`llm_model_path`), in Instruct mode
   (`llm_chat_template_kwargs`: `enable_thinking` off), with the sampling in
   `llm_sampling_params`. It runs text only, at 25% of the GPU
   (`llm_gpu_memory_utilization`) and the native 262,144-token context (ADR-013).
10. The summary goes to `bronze/ml-results/llm/<document_id>`.
11. The LLM publishes the result to `llm:tasks`.
12. The backend consumes the result and writes `silver` with its own grants.

Image tasks follow the same steps with one page. The worker converts the image
to RGB first. It raises an error when the OCR text is empty.

### 2. NER — spaCy mock (logs only)

1. Backend publishes `{document_id, text}` to `ner:tasks`.
2. The worker receives it through `ml-workers`.
3. On first run, `sync_if_missing()` downloads `models/ner/model-best/*` into
   `/model-cache/spacy/ner-model`. It does nothing when the directory is not
   empty. It does not verify checksums.
4. If `NER_MODEL_URI` is set, the worker loads the model from the MLflow
   registry instead.
5. spaCy extracts entities and the worker **logs them**. There is no result
   stream and no database write. This is the ADR-005 mock.

### 3. Embeddings — `gold.bo_embedding`

1. `scripts/backfill_embeddings.py` selects BOs from `silver.boletim` that
   have no `is_current` row in `gold.bo_embedding`. It is idempotent.
2. It publishes `{boletim_id, text, pipeline_run_id}` to `embedding:tasks`.
   It never writes the vector itself.
3. The embedding worker encodes with `neuralmind/bert-base-portuguese-cased`
   through sentence-transformers (768 dimensions).
4. In one transaction, it supersedes the previous current row, inserts the
   new row, and resolves or creates the `gold.model_registry` row.

The linkage classifier is calibrated against this encoder and its pooling.
Changing either one silently invalidates linkage scores.

### 4. Linkage scoring — `gold.bo_relacionado`

1. `scripts/backfill_linkage_scores.py` builds candidate pairs from two
   sources and removes duplicates:
   - HNSW top-K neighbours by embedding (default K = 20)
   - BOs that share a strong entity (person, document, plate, phone, bank
     account, …)
2. It publishes normalized `{id_bo_origem, id_bo_destino}` pairs to
   `linkage:tasks`.
3. The linkage worker loads both profiles on demand: current embedding,
   current entities, BO nature, and incident time. An in-memory cache keeps up
   to `linkage_profile_cache_max` (50,000) profiles. An evicted entry costs one
   re-encode and never changes a score.
4. It computes seven features and runs the calibrated LogisticRegression
   (`artifacts/bo_linkage_classifier_v1.joblib`). The features are: text
   cosine, geo (always 0 today), time decay, strong-id overlap, exact-location
   overlap, descriptive-text similarity, and entity-type profile.
5. It writes score, metrics, and `motivos` to `gold.bo_relacionado` with the
   same supersede-then-insert contract and `id_bo_origem < id_bo_destino`.

### 5. Clustering — `gold.clusters` / `gold.bo_cluster`

1. Any message in `cluster:tasks` is a trigger. It carries only
   `pipeline_run_id`.
2. The worker reads all current edges from `gold.bo_relacionado` with
   `score_sim >= 0.9` (`cluster_score_threshold`).
3. It builds a NetworkX graph and extracts connected components across all
   natures at once. This differs on purpose from the original per-nature
   algorithm.
4. In one transaction, it marks every previous cluster `is_current=false` and
   inserts the new batch.

Known data-quality issue: generic NER entities (`OBJETO_VEICULO`,
`DOCUMENTO`, …) create transitive bridges. In real data this produced 153
components. The largest held 4,634 of 5,013 nodes. The fix belongs upstream in
NER, not in the clustering layer.

### 6. BO maturity score — `scoring:tasks:results`

1. The producer sends `{boletim_id, ato_infracional, features,
   pipeline_run_id}` to `scoring:tasks`. `features` maps each flag to a
   boolean (victim, suspect, address, stolen items, …).
2. The worker reads the weights for that ato infracional from
   `artifacts/bo_scoring_weights_v1.json`. Weights for each nature sum to 1.0.
   It adds the weights of the present flags and applies the penalty rule for
   that nature:
   - homicídio without a victim: 10%
   - furto without the object and an address: 30%
   - every other nature: no penalty
3. It publishes `{boletim_id, ato_infracional, score, weights_version,
   pipeline_run_id}` to `scoring:tasks:results`.
4. The backend owns persistence. This service has no `gold.bo_score` table.

The worker needs only Redis. It has no database, MinIO, or GPU use.

The task carries the feature flags. Nothing derives them from the BO text yet.
A nature missing from the weights file raises an error and takes the normal
retry → DLQ path.

---

## Storage layout

| Location | Data of this service |
|---|---|
| MinIO `bronze` | source PDFs (backend), `ml-results/ocr/<id>`, `ml-results/llm/<id>` |
| MinIO `models` | NER spaCy directory (`ner/model-best`), synced to `/model-cache` |
| MinIO `mlflow-artifacts` | MLflow artifacts (core stack) |
| Postgres `silver` | read-only: `boletim`, and entity inputs for linkage |
| Postgres `gold` | `bo_embedding`, `bo_entidades` (read), `bo_relacionado`, `clusters`, `bo_cluster`, `model_registry` |
| `/model-cache` | Docker volume: vLLM and HuggingFace weights, spaCy model |
| `artifacts/` | `bo_linkage_classifier_v1.joblib`, `bo_scoring_weights_v1.json`, baked into the image |

---

## Known gaps

- **MLflow hostname.** The `config.py` default is `http://mlflow:5000`. The
  compose file has no MLflow service and sets no tracking URI. The optional
  registry path (`NER_MODEL_URI`, `OCR_MODEL_URI`) needs a reachable MLflow.
- **Embedding weights are not pinned to the volume.** `model_cache` is
  mounted but `HF_HOME` is not set. sentence-transformers can cache
  BERTimbau inside the container.
- **GPU placement is not in compose.** Compose reserves `count: 1` GPU for
  each GPU worker. The `*_GPU_DEVICE` variables and the 2-GPU routing in the
  README are set by the operator.
- **Scoring features are not extracted.** The scoring worker scores the
  `features` it receives. A producer must derive them.
- **NER entities have no worker path to the database.** The NER worker only
  logs. `gold.bo_entidades` is filled by the one-shot
  `scripts/import_bo_ner.py`.
- **`import_bo_ner.py` breaks the no-shared-code rule.** It adds the
  `sinapse` backend to `sys.path`, imports its models, and writes `silver`
  through the backend role. Treat it as a bootstrap exception, not a pattern.
- **Geo is always 0.** Nothing populates `silver.boletim.geom`, so `sim_geo`
  is 0.

---

## Design constraints

- **No shared Python package with the backend.** Stream names, table names,
  and payload shapes are contracts by convention.
- **Workers are stateless and scale horizontally.** The consumer group gives
  each message to one replica.
- **One GPU per heavy worker** (`ocr`, `llm`, `embedding`, `linkage`).
  `ner`, `cluster`, and `scoring` need no GPU. vLLM workers need
  `VLLM_WORKER_MULTIPROC_METHOD=spawn` because models load from a non-main
  thread.
- **Lazy model loading.** The first message pays the load cost. Models stay
  in memory until shutdown.
- **No internal HTTP between workers.** Each handoff is a stream. Each
  datastore write is its own transaction.
