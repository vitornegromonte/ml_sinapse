# ADR-002: Workers assíncronos com Redis Streams

**Status:** ✅ Aceito  
**Data:** 2026-06-02  

## Contexto

Os workers precisam consumir tarejas de forma desacoplada do backend FastAPI. O Redis já estava provisionado no `docker-compose.yml` mas não era utilizado.

## Decisão

Usar **Redis Streams** como mecanismo de fila entre o backend e os workers, com o padrão _consumer group_:

- Cada worker escuta uma stream dedicada: `ocr:tasks`, `ner:tasks`, `tokenizer:tasks`, `llm:tasks`
- Consumer group `ml-workers` garante distribuição round-robin se houver múltiplas réplicas do mesmo worker
- `XREADGROUP` com blocking evita polling ativo
- Dead-letter queue automática em `{stream}:dead` após `N` retentativas
- Tracking de retry via Redis Hash `{stream}:retries` com `hincrby`

```
Backend                  Redis                      Worker
  │                       │                          │
  ├─ XADD ocr:tasks ──────►                          │
  │                       │                          │
  │                       ├── XREADGROUP ────────────►
  │                       │                          ├── process()
  │                       │◄── XACK ─────────────────┤
  │                       │     (ou XADD :dead se    │
  │                       │      excedeu retries)    │
```

## Consequências

- **Positivas:** Desacoplamento total entre backend e ML; fila persistente em disco (Redis AOF/RDB); retry e DLQ nativos sem infra adicional; streams suportam múltiplos consumidores.
- **Negativas:** Dependência de Redis (mais um ponto de falha); latência adicional de rede; mensagens não são consumidas se o worker cair antes do ack (reprocessamento via `XCLAIM`).
- **Alternativas rejeitadas:** RabbitMQ (mais pesado), Celery (overhead de dependências), fila em PostgreSQL (mais latência e bloqueio de tabelas).
