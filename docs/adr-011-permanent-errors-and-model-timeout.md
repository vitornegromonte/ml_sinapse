# ADR-011: Falhas permanentes e timeout de modelo

**Status:** ✅ Aceito<br>
**Data:** 2026-11-19

## Contexto

O `BaseWorker` trata toda falha da mesma forma: tenta de novo até
`max_retries` (3), esperando `claim_idle_ms` (30 s) entre as tentativas, e só
então move a mensagem para `<stream>:dead`. Isso é certo para falhas
transitórias (rede, Redis, MinIO, memória da GPU). É um desperdício para uma
tarefa que nunca vai funcionar: um payload inválido ou um PDF acima do limite
de páginas ocupava o worker três vezes, e um PDF grande era processado três
vezes antes de ir para a DLQ.

Além disso, a chamada `predict()` do vLLM não tinha limite de tempo. Uma
chamada travada bloqueava o worker. Uma thread Python não pode ser
cancelada. Se o worker desistisse da chamada e continuasse, a próxima chamada
rodaria no mesmo engine vLLM enquanto a anterior ainda executa, e o engine não
é thread-safe.

## Decisão

1. **`PermanentTaskError`** (`workers/base_worker.py`, subclasse de
   `ValueError`). Quando `process()` o levanta, o `BaseWorker` copia a
   mensagem para `<stream>:dead` e faz o ack na primeira falha, sem passar pelo
   contador de retries. Só deve ser usado para tarefas que nunca terão
   sucesso. Não deve ser usado para falhas transitórias.
2. O worker de OCR levanta `PermanentTaskError` para: payload inválido ou sem
   fonte, PDF acima de `ocr_max_pdf_pages`, imagem ilegível ou corrompida. Um
   erro de rede ao baixar `image_url` continua sendo repetido. Texto vazio
   continua sendo repetido, pois pode ser uma falha do modelo.
3. **Timeout de modelo.** Cada chamada ao modelo usa
   `asyncio.wait_for(..., ocr_predict_timeout_seconds)`. Ao estourar, o worker
   levanta `OcrTimeoutError`, que segue o caminho normal de retry/DLQ (então um
   documento que sempre trava chega à DLQ após `max_retries` tentativas). O
   worker também para de ler mensagens. Ao sair do loop, `run()` não chama
   `unload()` e encerra o processo com `os._exit(1)`, porque a thread travada
   ainda usa o modelo e o interpretador esperaria por ela para sempre.
4. O `docker-compose.yml` passa a ter `restart: unless-stopped` no
   `ocr-worker`, para que um worker encerrado desse modo seja substituído.

## Consequências

- **Positivas:** tarefas impossíveis vão para a DLQ na hora, sem custo de
  GPU; um vLLM travado não bloqueia o worker para sempre nem roda em duplicidade.
- **Negativas:** a mensagem que estourou o tempo só é reprocessada depois que
  o worker reinicia e que `claim_idle_ms` passa; o modelo é recarregado a cada
  reinício, o que custa minutos de GPU. Um timeout muito baixo causa
  reinícios em série.
- **Obrigação:** só funciona com uma política de restart. Fora do compose
  (outro orquestrador), configure uma equivalente.
- **Alternativas rejeitadas:** continuar após o timeout (risco de duas chamadas
  no mesmo engine); rodar o modelo em um subprocesso que pode ser morto (mais
  complexo, fica para quando o timeout acontecer na prática).
- **Outros workers:** podem levantar `PermanentTaskError` sem mudar o
  `BaseWorker`. O timeout de modelo vale só para o OCR por enquanto.
