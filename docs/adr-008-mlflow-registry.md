# ADR-008: MLflow para experimentos e registry

**Status:** ✅ Aceito  
**Data:** 2026-06-02  

## Contexto

O time precisa versionar experimentos de treino, comparar métricas entre execuções e promover modelos de staging para produção de forma rastreável.

## Decisão

Adotar **MLflow** como plataforma de MLOps com dois componentes principais:

**1. Tracking Server** — serviço no docker-compose:
- Backend store: PostgreSQL (database `mlflow_db`)
- Artifact store: MinIO (bucket `mlflow-artifacts`)
- Porta: `5001` (host) → `5000` (container)

**2. Model Registry** — integrado ao Tracking Server:
- Workers carregam modelos via URI (`models:/nome/stage`)
- Fallback: se `{worker}_model_uri` vazio, usa path local (comportamento atual)

```
Treino                              Produção
  │                                   │
  ├─ mlflow.start_run()               │
  ├─ log_params/hp/artifacts          │
  ├─ log_model (transformers/spaCy)   │
  └─ register_model("ner-model")      │
        │                             │
        ▼                             │
  MLflow Registry              Worker (ner_worker.py)
  ├─ Staging                    ├─ cfg.ner_model_uri set?
  │─ Production ◄───────────────┤─ Sim → load_from_registry()
  └─ Archived                   └─ Não → spacy.load(path)
```

## Consequências

- **Positivas:** Rastreabilidade completa de experimentos → modelo → produção; workers com fallback hierárquico (registry > local); PostgreSQL como backend store reusa banco existente; MinIO como artifact store.
- **Negativas:** MLflow image oficial não tem `psycopg2` (Dockerfile customizado necessário); latência adicional no carregamento do modelo via registry; requer bucket `mlflow-artifacts` criado antes do server iniciar.
- **Alternativas rejeitadas:** SQLite como backend (não escala com múltiplos usuários), S3 direto sem MinIO (dependência de cloud), apenas paths locais (sem versionamento ou rastreabilidade).
