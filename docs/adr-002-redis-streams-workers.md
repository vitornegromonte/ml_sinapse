# ADR-002: Workers assíncronos com Redis Streams

**Status:** ✅ Aceito (atualizado em 2026-10-28)<br>
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

## Atualização (2026-10-28)

O corpo acima é histórico. O que **não** vale hoje:

- **Persistência não é garantida por este repo.** AOF/RDB é configuração do
  Redis do backend (`sinapse/backend/`), não do `ml_sinapse`; sem AOF, a PEL e
  o hash de retry são só RAM (B18).
- **Retry e DLQ não são nativos.** São implementados em
  `workers/base_worker.py`: hash `{stream}:retries`, DLQ `{stream}:dead` com
  `maxlen` + TTL, reclaim por `XAUTOCLAIM` e lease de heartbeat para não
  roubar mensagem em voo.
- **A lista de streams mudou:** `ocr:tasks`, `llm:tasks`, `ner:tasks`,
  `embedding:tasks`, `linkage:tasks`, `cluster:tasks`. `tokenizer:tasks` não
  existe.
- **Backpressure é consumer-group lag** (`XINFO GROUPS`), não `XLEN`; requer
  Redis ≥ 7.
- **A rejeição de fila em PostgreSQL está desatualizada:**
  `SELECT ... FOR UPDATE SKIP LOCKED` resolve o bloqueio citado. A decisão
  está em B19, não neste ADR.
