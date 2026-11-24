# ADR-015: Observabilidade dos workers: métricas, saúde e logs estruturados

**Status:** ✅ Aceito<br>
**Data:** 2026-11-24

## Contexto

Os workers só escreviam logs de texto. A revisão (`ML_review.md`, F8) apontou
que a fila, as mensagens mortas, a latência e o estado do modelo eram invisíveis:
um worker travado, uma DLQ crescendo ou o corte do stream de resultados só
apareciam quando o consumidor reclamava. O ADR-014 já põe um `manifest_id` em
cada saída; faltava ver isso em operação.

## Decisão

1. **Métricas Prometheus** (`services/telemetry.py`), por processo, com um
   registro próprio:
   - `ml_messages_total{stream,outcome}`, com `outcome` em `success`, `retry`,
     `dead_letter` (retries esgotados) e `permanent` (`PermanentTaskError`);
   - `ml_messages_reclaimed_total`, `ml_process_duration_seconds` e
     `ml_batch_size` (uma observação por chamada de `process_batch`);
   - `ml_queue_length`, `ml_queue_pending` e `ml_dead_letter_length`, atualizados
     a cada `METRICS_REFRESH_SECONDS` (15) por uma tarefa própria;
   - `ml_result_stream_length`, para ver o stream de resultados chegar ao teto;
   - `ml_model_loaded` e `ml_model_info{manifest_id,model}` (valor 1), que
     mostra a receita que cada worker roda;
   - `ml_last_message_timestamp_seconds`.
   Os rótulos são só o stream, o desfecho e a receita: nenhum id de documento,
   nenhum texto.
2. **Endpoints** num servidor HTTP mínimo em uma thread, só quando
   `METRICS_PORT` está definido (0 desliga, é o padrão):
   - `/metrics` (formato Prometheus);
   - `/healthz`: 200 enquanto o laço do worker faz progresso. O laço principal e
     o heartbeat do lease contam, então uma chamada longa ao modelo não derruba
     a saúde, e um event loop bloqueado derruba. O limite é
     `POLL_TIMEOUT_MS/1000 + HEALTH_GRACE_SECONDS` (80 s);
   - `/readyz`: 200 enquanto o laço roda **e** o Redis respondeu há pouco. O
     modelo não entra: os modelos carregam na primeira tarefa, e um worker
     ocioso nunca ficaria pronto. Veja `ml_model_loaded`.
3. **A observabilidade nunca atrapalha o processamento.** Todo método de
   gravação captura o próprio erro. Com telemetria desligada, todos os ganchos
   do `BaseWorker` são no-ops e o caminho de ack, retry e DLQ é o mesmo.
   Uma porta ocupada é registrada e ignorada.
4. **Logs estruturados** (`services/logging_setup.py`). `LOG_FORMAT=text` (padrão)
   mantém o formato de sempre. Com `LOG_FORMAT=json`, cada linha é um objeto com
   `ts`, `level`, `logger`, `message`, `stream`, `manifest_id`, `model` e o
   contexto da mensagem: `message_id` e, quando existem no payload,
   `pipeline_run_id`, `document_id`, `boletim_id`, `id_bo_origem` e
   `id_bo_destino`. Um lote com várias mensagens leva só `message_ids`. O
   contexto segue para tarefas asyncio e para threads (`asyncio.to_thread`).
   Só ids entram, nunca o texto.
5. **Compose**: `METRICS_PORT=9100` e um healthcheck compartilhado (`/healthz`,
   `start_period` de 300 s para o primeiro download e carga do modelo) em todos
   os workers. A porta não é publicada no host.

## Medições

Teste de ponta a ponta com um processo real do worker de scoring e um Redis real
(6.2, embutido), cinco tarefas: três válidas, uma com `features` em texto JSON e
uma com natureza desconhecida.

| Medida | Resultado |
|---|---|
| `/healthz` e `/readyz` com o worker rodando | 200 e 200 |
| Métricas lidas do processo | 3 `success`, 2 `dead_letter`, DLQ com 2, stream de resultados com 3, 5 chamadas observadas |
| O que o Redis tinha | 3 resultados e 2 mensagens mortas (bate com as métricas) |
| `ml_model_info` | o `manifest_id` do manifesto de scoring |
| Logs | 22 linhas, todas JSON, com `message_id`, `pipeline_run_id`, `boletim_id` e `manifest_id` |
| SIGTERM | saída 0 e a porta fechada |

## Consequências

- **Positivas:** fila, DLQ, latência, desfecho e receita ficam visíveis e
  alertáveis, e uma busca por `document_id` ou `pipeline_run_id` segue um
  documento entre os workers.
- **O teste achou um problema de contrato, fora do escopo daqui.** O
  `ScoringTask.features` é um `dict`, mas um campo de stream do Redis é uma
  string. Um texto JSON é rejeitado e vai para a DLQ. A menos que o produtor do
  backend codifique `features` de outro jeito, o worker não recebe features.
  **Confirme a codificação do produtor.**
- **Os gauges da fila têm até `METRICS_REFRESH_SECONDS` de atraso.**
- **O healthcheck do compose só marca o contêiner como `unhealthy`.** Reiniciar
  exige o orquestrador ou uma ferramenta como o autoheal.
- **Não coberto:** alertas e painéis (esta mudança só expõe os dados), métricas
  de GPU (use o exportador da NVIDIA), a contagem de páginas OCR cortadas pelo
  limite de tokens, o rastreamento distribuído e a medição em uma GPU real. Os
  testes usam mocks e o teste de ponta a ponta usou o worker de scoring, que não
  carrega modelo.
