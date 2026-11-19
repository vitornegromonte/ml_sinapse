from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    redis_url: str = "redis://redis:6379/0"

    # Least-privilege `sinapse_ml` role (member of `ml_worker`: read silver,
    # write gold — never silver). See backend/alembic/versions/0002_schema_
    # roles_and_grants.py and 0004_add_ml_login_role.py.
    ml_database_url: str = "postgresql+asyncpg://sinapse_ml:change_me_ml_worker@db:5432/sinapse_db"

    model_base_path: str = "/model-cache"

    # Default production OCR model: Dharma-AI/Dharma-OCR-LITE (Qwen2.5-VL
    # fine-tune). It replaced zai-org/GLM-OCR, which scored 85.1 Overall on the
    # Portuguese MDPBench samples against Dharma's 56.0 — GLM-OCR is still
    # registered and selectable, see models/ocr/registry.py and ADR-004.
    ocr_model_name: str = "Dharma-AI/Dharma-OCR-LITE"
    # DPI used to render scanned PDF pages and image blocks before OCR. Bounded
    # on purpose: image tokens grow with the square of the DPI, so an unchecked
    # value can exceed max_model_len or the GPU budget. At 200 dpi an A4 page is
    # about 1654x2339 px, roughly 4.9k Qwen2.5-VL image tokens.
    ocr_render_dpi: int = Field(default=200, ge=72, le=400)
    ocr_gpu_device: int = 0
    # Output cap per page. 2048 can truncate dense pages and tables; the
    # input side of a 200 dpi A4 page is about 4.9k tokens, so 4096 output
    # tokens still fits the model's native context.
    ocr_max_tokens: int = 4096
    ocr_request_timeout: int = 120  # seconds
    # Scanned pages sent to the model in one call, so vLLM batches them on the
    # GPU. Measured on 12 scanned pages (Dharma-OCR-LITE, RTX PRO 6000, 15%
    # cap, warm model): batch 1 = 248 s, 2 = 141 s, 4 = 83 s (3.0x), 8 = 65 s
    # (3.8x). 4 is the default because batch 8 left only ~0.5 GiB under the 15%
    # GPU cap. Use 1 for one page per model pass (OCR_PAGE_BATCH_SIZE=1). A
    # failed batch costs less now: finished batches are checkpointed.
    ocr_page_batch_size: int = Field(default=4, ge=1, le=32)
    # Max seconds for one model call (one batch of pages). A Python thread
    # cannot be cancelled, so a timeout retires the worker process instead of
    # reusing a model that may still be running — see OcrWorker._predict.
    ocr_predict_timeout_seconds: int = Field(default=600, ge=1)
    # Finished scanned pages of a PDF are kept in Redis for this long, so a
    # retry of the same task resumes instead of OCR-ing every page again. 0
    # turns the checkpoint off. See services/ocr_checkpoint.py.
    ocr_partial_ttl_seconds: int = Field(default=86400, ge=0)
    # Local images (a task's `image_path`) are read only from inside this
    # directory — see services/image_loader.py. Empty disables local-path
    # reads: without a root, image_path is an arbitrary-file-read vector.
    ocr_image_allowed_root: str = ""
    # Result streams hold OCR/LLM output until the backend consumer drains it.
    # The cap is a RAM valve for a dead consumer, not a queue size: whatever
    # overflows it is dropped with no error. At 1000 a 10k-PDF backfill could
    # trim the oldest results while the consumer lagged. 50k leaves a full
    # backfill of headroom, and ResultPublisher logs when the cap actually
    # bites (that log is the real signal — raise the cap or drain the stream).
    ocr_result_stream_maxlen: int = 50000
    # Safety cap for PDF tasks: a pathological document (or a corrupt/hostile
    # upload) shouldn't monopolize a worker indefinitely. Exceeding this raises
    # in _process_pdf, which feeds the existing retry/DLQ path in base_worker.py
    # rather than hanging the consumer loop.
    ocr_max_pdf_pages: int = 200
    # vLLM tuning (any VllmChatOcrModel subclass — see models/ocr/vllm_chat_base.py).
    # Mirrors the llm_* naming already used for the LLM worker's own vLLM engine.
    ocr_dtype: str = "auto"
    # Share of GPU memory the OCR vLLM engine may use. Capped at 15% on purpose:
    # the card is shared with the embedding and linkage workers (and, on shared
    # hosts, with other jobs). Dharma-OCR-LITE needs ~4.6 GiB of weights, and
    # its KV cache is ~36 KiB per token, so 15% of a 95 GiB card still holds
    # the model's full 128k context. Values above 0.15 are rejected.
    ocr_gpu_memory_utilization: float = Field(default=0.15, gt=0, le=0.15)
    # Context cap for the OCR engine. A page needs at most 16,384 image tokens
    # (the Qwen2.5-VL processor's max_pixels) + max_tokens output + the prompt,
    # which is under 24,576. The model's native 128k only inflates vLLM's
    # memory profiling: at 15% of the GPU it does not even start.
    ocr_max_model_len: int | None = 24576

    # sentence-transformers encoder used to calibrate the linkage classifier
    # (Maynara_cluster_v0/modelo_v1_classificador.joblib) — MUST stay this
    # exact model + pooling. Swapping the encoder changes the embedding space
    # and silently invalidates the classifier's calibration without raising
    # any error. See models/embedding_model.py's SentenceTransformerEncoder docstring.
    embedding_model_name: str = "neuralmind/bert-base-portuguese-cased"
    embedding_gpu_device: int = 0

    # Classificador calibrado do motor de linkage (Maynara_cluster_v0) — ver
    # workers/linkage_scoring_worker.py. Roda na mesma GPU do
    # embedding worker (uso leve, CPU já seria suficiente pro classificador
    # em si; o que pesa é reencodar descrições, mesmo encoder do embedding).
    linkage_classifier_path: str = "/app/artifacts/bo_linkage_classifier_v1.joblib"
    linkage_gpu_device: int = 0
    linkage_tau_dias: float = 30.0
    linkage_limiar_descricao: float = 0.6
    # Perfis de BO mantidos em memória no worker de linkage (embedding + vetores
    # das descrições, ~15-40 KB por BO). 490k pares reusam ~50k BOs; sem cache
    # cada BO é re-encodado na GPU ~20x. Uma entrada evictada só custa um
    # re-encode — nunca um score diferente. Ver workers/linkage_scoring_worker.py.
    linkage_profile_cache_max: int = 50_000

    # "Parâmetros calibrados nesta versão" do README do Maynara_cluster_v0 —
    # não o default 0.5 do código-fonte de clusters_por_natureza() (os dois
    # divergem no pacote original; este é o valor que o README chama de
    # validado). Ver workers/cluster_worker.py.
    cluster_score_threshold: float = 0.9

    ner_model_path: str = "/model-cache/spacy/ner-model"
    # Where the spaCy model directory lives under the `models` bucket when it's
    # raw files (not MLflow-registered). GUESS based on ADR-005's `model-best/`
    # naming — verify with `mc ls sinapse/models/` and override if different.
    # Only used as a fallback when ner_model_uri is unset.
    ner_model_minio_prefix: str = "ner/model-best"

    llm_model_path: str = ""
    llm_gpu_device: int = 0
    llm_gpu_memory_utilization: float = 0.9
    llm_max_model_len: int | None = None
    llm_tensor_parallel_size: int = 1
    llm_dtype: str = "auto"
    llm_sampling_params: dict | None = None
    llm_result_stream_maxlen: int = 50000
    llm_prompt_template: str = (
        "Você é um assistente especializado em analisar documentos policiais.\n"
        "Abaixo está o texto extraído de um documento:\n\n"
        "{text}\n\n"
        "Produza um resumo conciso destacando:\n"
        "1. Fatos principais\n"
        "2. Envolvidos (se houver)\n"
        "3. Datas e locais mencionados\n"
        "4. Provas documentais citadas\n\n"
        "Resumo:"
    )

    ocr_stream: str = "ocr:tasks"
    ner_stream: str = "ner:tasks"
    llm_stream: str = "llm:tasks"
    embedding_stream: str = "embedding:tasks"
    linkage_stream: str = "linkage:tasks"
    cluster_stream: str = "cluster:tasks"
    scoring_stream: str = "scoring:tasks"
    # Weights per ato infracional for the BO maturity score. Data, not code:
    # recalibrating a nature is a file change, not a deploy of new logic (the
    # two penalty rules are the exception — they live in occurrence_scoring.py).
    scoring_weights_path: str = "artifacts/bo_scoring_weights_v1.json"
    scoring_result_stream_maxlen: int = 50000
    dead_letter_suffix: str = ":dead"
    # Bound the dead-letter stream: without this it grows forever (every
    # permanently failing message is appended and nothing ever reads it).
    dead_letter_maxlen: int = 10000
    dead_letter_ttl_seconds: int = 2592000  # 30 days

    consumer_group: str = "ml-workers"
    consumer_id: str = ""

    max_retries: int = 3
    poll_timeout_ms: int = 50000
    # Messages read per XREADGROUP. Honoured only by workers with
    # supports_batch = True (embedding, llm), which turn the whole batch into
    # one encode()/generate() call; serial workers (ocr, ner, cluster,
    # linkage) keep reading one at a time so they never hold messages their
    # replicas could be working on.
    batch_size: int = 8
    # A message failed but hasn't hit max_retries is left un-acked (still
    # pending) rather than dropped — this is how long it sits idle before
    # XAUTOCLAIM reclaims it for another attempt. See workers/base_worker.py.
    claim_idle_ms: int = 30000
    # While a message is in flight the worker heartbeats a short-TTL lease so
    # another replica's XAUTOCLAIM sweep doesn't steal work that is still
    # running — claim_idle_ms alone can't tell "slow" from "crashed".
    lease_heartbeat_seconds: int = 10
    lease_ttl_seconds: int = 30

    log_level: str = "INFO"

    minio_endpoint: str = "http://minio:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket_bronze: str = "bronze"
    minio_bucket_models: str = "models"

    ner_model_uri: str = ""
    ocr_model_uri: str = ""


settings = Settings()
