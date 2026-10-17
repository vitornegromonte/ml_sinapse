# ADR-009: Wrappers de modelo modulares (BaseModelWrapper)

**Status:** ✅ Aceito  
**Data:** 2026-06-02  

## Contexto

O `ml_service` precisa suportar múltiplos backends de inferência (vLLM, transformers, llama.cpp, BentoML, MLflow) sem acoplar a lógica do worker ao framework específico.

## Decisão

Criar uma hierarquia de classes wrapper com `BaseModelWrapper` (ABC) como interface comum:

```
BaseModelWrapper (ABC)
├── load()
├── predict(inputs, **kwargs)
└── unload()

       │
       ▼
    LLMModel
    ├── engine="vllm"        → vLLM (GPU, alto throughput)
    ├── engine="transformers" → HF AutoModel (fallback)
    ├── engine="llama.cpp"   → llama.cpp (CPU, quantizado)
    ├── engine="bentoml"     → BentoML Runner
    └── engine="mlflow"      → MLflow pyfunc
```

Além disso, `BaseModelWrapper` expõe métodos estáticos auxiliares:

- `resolve_model_uri(uri, default_path, flavor)` — decide entre registry e path local
- `load_from_registry(uri)` — carrega do MLflow Model Registry
- `register_model(path, name, run_id)` — registra modelo no MLflow

## Consequências

- **Positivas:** Workers não precisam conhecer o engine específico; troca de engine via config (sem mudança de código); interface `load/predict/unload` padronizada; reúso em workers e no BentoML service.
- **Negativas:** Camada extra de abstração; predict precisa lidar com formatos de entrada/saída diferentes entre engines; `unload()` nem sempre é trivial (especialmente com CUDA).
