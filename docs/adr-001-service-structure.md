# ADR-001: Estrutura do serviço de ML

**Status:** ✅ Aceito  
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
