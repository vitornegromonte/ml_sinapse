# ADR-012: Velocidade do OCR e retomada após falha

**Status:** ✅ Aceito<br>
**Data:** 2026-11-19

## Contexto

O OCR de um PDF digitalizado era lento por três motivos, e uma falha perto do
fim perdia todo o trabalho:

- algumas páginas entravam em loop até o limite de `max_tokens` (4 096) e
  custavam cerca de 80 s cada;
- cada página ia sozinha para o modelo, e o decode de uma única sequência não
  usa a GPU inteira;
- um retry (ADR-011: até 3 tentativas) refazia o OCR de todas as páginas.

## Decisão

1. **`repetition_penalty` 1,05 no Dharma.** É o valor do `generation_config.json`
   do próprio modelo. Nosso `SamplingParams` explícito o ignorava. Só o
   `DharmaOcrModel` o usa (`REPETITION_PENALTY`); o GLM-OCR não muda.
2. **Batching de páginas.** `ocr_page_batch_size` tem padrão 4. O vLLM
   processa as páginas do lote juntas.
3. **Sobreposição de CPU e GPU.** Enquanto um lote está no modelo, o worker
   renderiza e corrige a inclinação das páginas do próximo lote, no mesmo laço
   (sem fila e sem tarefa separada). No máximo um lote está no modelo e um está
   sendo preparado. Se a preparação falha, o worker espera a chamada ao modelo
   terminar antes de propagar o erro.
4. **Checkpoint de páginas no Redis.** Ao fim de cada lote, os textos das
   páginas vão para o hash `<stream>:partial:<document_id>:<hash>`, com TTL de
   `ocr_partial_ttl_seconds` (86 400 s; 0 desliga). O hash cobre os bytes do
   PDF, o modelo, o DPI e `ocr_max_tokens`, então uma mudança em qualquer um
   deles usa outra chave. Num retry, o `PdfExtractor` recebe os números das
   páginas salvas (`cached_images`) e não as renderiza. O checkpoint só é apagado
   depois que o texto foi salvo no bronze e publicado. Falhas de Redis no
   checkpoint são registradas e ignoradas: no pior caso o OCR é refeito.
5. **Cache de compilação do vLLM.** O `ocr-worker` usa o volume `vllm_cache`
   (`VLLM_CACHE_ROOT=/vllm-cache`), para que o reinício do ADR-011 não compile
   os kernels de novo.

## Medições

GPU RTX PRO 6000 (95 GiB, compartilhada), Dharma-OCR-LITE, 15% de memória,
200 dpi, modelo aquecido. Os PDFs de teste não são BOs.

| Medida | Resultado |
|---|---|
| 12 páginas digitalizadas, lote 1 / 2 / 4 / 8 | 248 s / 141 s / 83 s / 65 s (velocidade contra o lote 1: 1,8× / 3,0× / 3,8× nos lotes 2 / 4 / 8) |
| `doc.pdf`, 5 imagens, penalidade 1,0 | 2 páginas no limite de tokens nas 2 repetições; 83 s |
| `doc.pdf`, 5 imagens, penalidade 1,05 | 0 páginas no limite; 15 s e 22 s |
| Renderizar e corrigir 12 páginas (só CPU) | 1,1 s (0,09 s por página) |
| Retomada: falha na 3ª chamada (lote 2), 12 páginas | 4 páginas salvas; o retry enviou 8 de 12 ao modelo |
| Carga do motor vLLM, cache frio / quente | 137 s / 73 s |

A saída muda um pouco entre execuções em lote (o tamanho do lote altera a
numérica), então não se deve esperar texto idêntico byte a byte.

## Consequências

- **Positivas:** o lote de 4 deu ~3× de velocidade; a penalidade removeu as
  páginas em loop no arquivo onde elas apareciam; um retry não repete páginas
  prontas; o reinício após timeout é mais rápido.
- **A sobreposição quase não ajuda aqui.** O preparo das páginas custa 0,09 s
  contra ~17 s por página no modelo, então ela esconde menos de 1% do tempo.
  Ela passa a importar se o preparo ficar caro (DPI alto, páginas grandes).
- **Lote 8 foi mais rápido (3,8×),** mas o pico de memória da GPU ficou cerca de
  0,5 GiB abaixo do teto de 15%. Por isso o padrão é 4.
- **Lote maior atrasa o primeiro resultado** e, se uma chamada falha, repete
  mais páginas (limitado pelo checkpoint).
- **Memória no Redis:** cada PDF em andamento guarda o texto de suas páginas
  digitalizadas por até 24 h. O custo é pequeno (KB por página), e o TTL limita.
- **Não medido:** a penalidade em BOs reais; o efeito de DPI e `max_tokens`
  sobre a velocidade; vários workers na mesma GPU.
