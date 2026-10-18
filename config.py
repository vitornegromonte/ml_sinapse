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

    # Default production OCR model — benchmarked
    # best of 8 models tested against Portuguese MDPBench samples (Overall
    # 85.1). Was Dharma-AI/Dharma-OCR-LITE (scored 56.0, lowest of the 8) —
    # still registered/selectable, see models/ocr/registry.py.
    ocr_model_name: str = "zai-org/GLM-OCR"
    ocr_gpu_device: int = 0
    ocr_max_tokens: int = 2048
    ocr_request_timeout: int = 120  # seconds
    ocr_result_stream_maxlen: int = 1000
    # Safety cap for PDF tasks: a pathological document (or a corrupt/hostile
    # upload) shouldn't monopolize a worker indefinitely. Exceeding this raises
    # in _process_pdf, which feeds the existing retry/DLQ path in base_worker.py
    # rather than hanging the consumer loop.
    ocr_max_pdf_pages: int = 200
    # vLLM tuning (any VllmChatOcrModel subclass — see models/ocr/vllm_chat_base.py).
    # Mirrors the llm_* naming already used for the LLM worker's own vLLM engine.
    ocr_dtype: str = "auto"
    ocr_gpu_memory_utilization: float = 0.9
    ocr_max_model_len: int | None = None

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
    llm_result_stream_maxlen: int = 1000
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
    dead_letter_suffix: str = ":dead"
    # Bound the dead-letter stream: without this it grows forever (every
    # permanently failing message is appended and nothing ever reads it).
    dead_letter_maxlen: int = 10000
    dead_letter_ttl_seconds: int = 2592000  # 30 days

    consumer_group: str = "ml-workers"
    consumer_id: str = ""

    max_retries: int = 3
    poll_timeout_ms: int = 50000
    batch_size: int = 1
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
