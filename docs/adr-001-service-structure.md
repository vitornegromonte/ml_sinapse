# ADR-001: Estrutura do serviço de ML

**Status:** ✅ Aceito (histórico — ver Atualização)<br>
**Data:** 2026-06-02  

## Contexto

O projeto precisa de um subsistema de ML modular que atenda a múltiplas tarefas (OCR, NER, tokenização/embeddings, LLM) de forma isolada e escalável.

## Decisão

Criar `ml_service/` com a seguinte estrutura:

```
ml_service/
├── __init__.py
├── config.py              # Config centralizada (pydantic-settings)
├── requirements.txt
├── Dockerfile
├── models/                # Wrappers de modelos
│   ├── base_model.py      # Abstract BaseModelWrapper
│   ├── llm_model.py       # LLM com engines vLLM/transformers/llama.cpp/BentoML/MLflow
│   └── bento.py           # Definição BentoML service
├── workers/               # Workers assíncronos
│   ├── base_worker.py     # Loop Redis Streams + DLQ + retry
│   ├── ocr_worker.py      # OCR via VLM (HuggingFace)
│   ├── ner_worker.py      # NER mock com spaCy (provisório)
│   └── tokenizer_worker.py# Tokenização + embeddings BERTimbau
├── scripts/
│   └── init_buckets.sh    # Inicialização de buckets MinIO
└── training/              # Scripts de treino (futuro)
```

## Consequências

- **Positivas:** Separação clara entre modelos, workers e config; cada worker é um processo independente; config centralizada via env vars.
- **Negativas:** Workers não podem compartilhar GPU diretamente (cada processo precisa da sua alocação); complexidade adicional de orquestração via Redis.
- **Neutro:** O diretório `models/` serve tanto para uso interno dos workers quanto para deploy via BentoML.

## Atualização (2026-10-28)

O corpo acima é histórico. O pacote foi achatado de `ml_service/` para a raiz
do repo e a árvore atual é:

```
./
├── config.py
├── run_worker.py          # um worker por processo/container
├── workers/               # base, ocr, llm, ner, embedding, linkage_scoring, cluster
├── models/                # base_model, embedding_model, llm_model, tasks, ocr/
├── services/              # storage, database, gold, image_loader, publishers, normalização
└── scripts/
```

- `models/bento.py` foi removido em `fc49c1c`.
- `workers/tokenizer_worker.py` foi removido em B5 (`4997308`); não há worker
  nem stream de tokenização.
- `training/` nunca existiu neste repo.
- `LLMModel` hoje é vLLM-only; `BaseModelWrapper` é só a ABC mínima
  (`load`/`predict`/`unload`), não o dispatch de engines descrito aqui.
