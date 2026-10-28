# ADR-010: Serving com BentoML

**Status:** ⛔ Substituído (2026-10-28) — serviço removido em `fc49c1c`<br>
**Data:** 2026-06-02  

## Contexto

Além da inferência via workers (Redis Streams), o modelo de LLM pode precisar ser servido como API REST para consumo síncrono (ex: agente conversacional, testes interativos). Também pode ser necessário fazer deploy do modelo em ambientes externos sem depender da infraestrutura completa de workers.

## Decisão

Usar **BentoML** como framework de serving complementar:

- Serviço definido em `ml_service/models/bento.py` com anotação `@bentoml.service`
- Reutiliza o `LLMModel` existente (qualquer engine)
- Expõe duas APIs:
  - `generate(prompt)` — inferência single
  - `generate_batch(prompts)` — inferência em lote (batchable, max 32)
- Pode ser servido standalone: `bentoml serve ml_service.models.bento:LLMBentoService`
- Ou buildado como Bento: `bentoml build` → container OTEL-ready
- GPU configurada via `resources={"gpu": 1}` no decorator

## Consequências

- **Positivas:** Serving REST sem depender dos workers; BentoML gerencia OTEL, health checks, metrics Prometheus; deployável em qualquer ambiente (Docker, K8s, BentoCloud); reúso do `LLMModel` existente.
- **Negativas:** Mais um framework na stack; BentoML v1.x está em evolução rápida (API changes entre versões); não substitui os workers (que continuam sendo o pipeline principal para processamento assíncrono).
- **Relação com workers:** Workers são o pipeline principal (OCR → NER → tokenizer). BentoML é uma alternativa para serving síncrono do LLM ou para deploy isolado.

## Atualização (2026-10-28)

Este ADR está **substituído**. `models/bento.py` e o serviço BentoML foram
removidos em `fc49c1c`, quando o `LLMModel` foi cortado para vLLM-only. Não há
serving REST neste repo. Se a necessidade voltar, abrir um ADR novo — este
desenho depende de um `LLMModel` multi-engine que não existe mais.
