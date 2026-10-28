# ADR-003: Isolamento de workers via multiprocessing

**Status:** ✅ Aceito (atualizado em 2026-10-28)<br>
**Data:** 2026-06-02  

## Contexto

Os workers (OCR, NER, tokenizer, LLM) podem usar GPUs diferentes e bibliotecas que não liberam o GIL (PyTorch, TensorFlow). É necessário decidir como executá-los concorrentemente.

## Decisão

Usar `multiprocessing.Process` no `main.py`, onde cada worker roda em seu próprio processo com seu próprio event loop `asyncio`:

```
main.py
  ├── Process(target=OcrWorker().run)
  ├── Process(target=NerWorker().run)
  ├── Process(target=TokenizerWorker().run)
  └── Process(target=LlmWorker().run)
```

Cada worker:
- Tem seu próprio event loop (`asyncio.new_event_loop()`)
- Tem sua própria conexão Redis
- Pode ser alocado a uma GPU diferente via env vars (`CUDA_VISIBLE_DEVICES` ou config `ocr_gpu_device`)
- Trata `SIGTERM`/`SIGINT` para shutdown gracefull

## Consequências

- **Positivas:** Isolamento total de memória GPU; um worker não derruba os outros; cada worker pode usar uma versão diferente de framework; `main.py` é o entrypoint único no Docker.
- **Negativas:** Consumo maior de RAM (cada worker carrega seu modelo); comunicação entre workers precisa ser via Redis (não memória compartilhada); `multiprocessing` com GPU pode exigir `spawn` como start method.
- **Alternativas rejeitadas:** `asyncio.gather` (GIL bloqueia CPU-bound), threads (`torch` não libera GIL), containers separados (complexidade adicional de orquestração sem Kubernetes).

## Atualização (2026-10-28)

O `main.py` com `multiprocessing.Process` **não existe mais**. Cada worker é
seu próprio processo/container, iniciado por `python run_worker.py <nome>`;
`run_worker.py` resolve só o worker pedido via `importlib` (B11), sem importar
os outros cinco.

O isolamento de memória/GPU continua, mas no nível de container, não de
`Process` dentro de um entrypoint único. `TokenizerWorker` foi removido (B5).
O `spawn` citado continua necessário no vLLM, mas por outro motivo: o engine
core não pode ser forkado de uma thread que já tocou CUDA
(`VLLM_WORKER_MULTIPROC_METHOD=spawn`).
