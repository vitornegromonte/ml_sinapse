# ADR-013: LLM de resumo — maritaca-ai/S5-nano

**Status:** ✅ Aceito<br>
**Data:** 2026-11-24

## Contexto

O worker `llm` resume o texto do OCR. O modelo anterior era
`google/gemma-4-E4B-it`, escolhido no código só como valor padrão de um script,
e `LLM_MODEL_PATH` vinha vazio no compose. O pedido foi trocar para
`maritaca-ai/S5-nano`: modelo de 4B parâmetros da Maritaca AI, sobre o
Qwen3.5-4B, licença Apache-2.0, português e inglês, contexto de 262 144 tokens.

Três fatos sobre o modelo e sobre o código mudaram a troca de "mudar um nome"
para uma mudança de configuração:

- **O S5-nano começa em modo Think**: escreve o raciocínio antes da resposta.
  Um resumo não pode conter isso.
- **O worker nunca definiu `max_tokens`.** O `SamplingParams()` do vLLM usa
  `max_tokens=16` e temperatura 1,0. Sem `LLM_SAMPLING_PARAMS` no ambiente (o
  compose não o define), todo resumo era cortado em cerca de 16 tokens. Só os
  scripts passavam `max_tokens`, e os testes usam mocks.
- **O S5-nano é um modelo híbrido** (atenção linear em 24 das 32 camadas) com
  encoder de visão. O vLLM 0.22.1, fixado em `requirements.txt`, não inicia com
  o `max_num_seqs` padrão de 1024 nessa memória (`max_num_seqs (1024) exceeds
  available Mamba cache blocks (713)`).

## Decisão

1. `llm_model_path` tem padrão `maritaca-ai/S5-nano` (o compose também). A
   versão BF16, como pedido. A versão FP8 (`maritaca-ai/S5-nano-FP8`, 5,8 GB) é
   recomendada pelo fabricante para placas com FP8 nativo e basta trocar a
   variável; não foi testada aqui.
2. **Modo Instruct**: `llm_chat_template_kwargs = {"enable_thinking": False}`,
   passado ao `apply_chat_template`. Um modelo cujo template não usa a variável
   a ignora.
3. **Amostragem** (`llm_sampling_params`): os valores que o fabricante
   recomenda para o modo Instruct (temperatura 0,7, `top_p` 0,8, `top_k` 20,
   `min_p` 0, `presence_penalty` 1,5) mais `max_tokens` 2048. O `LLMModel`
   avisa no log quando `max_tokens` falta.
4. **Argumentos do motor** (`llm_engine_kwargs`): `language_model_only` (sem o
   encoder de visão) e `max_num_seqs` 32 (o worker lê 8 mensagens por vez).
5. **Memória**: `llm_gpu_memory_utilization` passou de 0,9 para 0,25 e o
   contexto fica nativo (262 144).
6. No compose, `HF_HOME=/model-cache/huggingface`, para que o download de ~9 GB
   de um id do Hugging Face sobreviva ao reinício do contêiner.

`LLMModel` ganhou os parâmetros `chat_template_kwargs` e `engine_kwargs`. A
escolha do modelo e de seu modo fica na configuração, não na classe.

## Medições

GPU RTX PRO 6000 (95 GiB, compartilhada), vLLM **0.22.1** e transformers 5.14.1
(as versões fixadas), 25% da placa, contexto nativo. Entrada: um BO
**fictício** em português, escrito para o teste (6,7 mil caracteres).

| Medida | Resultado |
|---|---|
| Pesos (só texto) | 7,99 GiB |
| Cache KV | 13,6 GiB, 442 mil tokens (1,69× o contexto cheio) |
| Memória da placa | +23,5 GB depois de carregar e de gerar um resumo; +26 GB no fim do teste (provavelmente inclui o contexto CUDA do processo de teste) |
| Carga mais o primeiro resumo | 237 s com o cache de compilação vazio (`torch.compile`: 44 s); 151 s com o cache cheio (`torch.compile`: 3 s). A fase de perfil da memória (75 a 83 s) não depende do cache |
| Um resumo | 25,8 s, 587 tokens (22,8 tokens/s) |
| Lote de 8 resumos | 34,5 s, 4 657 tokens (135 tokens/s) |
| Comportamento antigo (sem `max_tokens`, modo Think) | 16 tokens: "We need answer in Portuguese. User asks produce concise summary…" |
| Modo Think com `max_tokens` 700 | o texto começa com raciocínio em inglês ("We need answer in Portuguese…"); não inspecionei o fim |
| Novo padrão | resumo em 4 seções, em português, sem raciocínio. Manteve os nomes, os horários e os números de documentos (NF-e, laudo, placa). Omitiu dois valores do texto: R$ 12.800,00 (notebooks) e R$ 3.000,00 (papel). Um só texto, lido por mim |

## Consequências

- **Positivas:** os resumos deixam de ser cortados em 16 tokens e de começar
  pelo raciocínio do modelo; o contexto de 262 mil tokens cobre PDFs longos;
  a placa fica com mais memória livre do que com 0,9.
- **`presence_penalty` 1,5** evita laços de repetição, mas também desencoraja
  reaproveitar nomes e números. O teste acima não mostrou perda, mas só usou um
  texto fictício. Avalie em BOs reais antes de confiar nele.
- **Qualidade não medida**: não houve comparação com o Gemma, nem com BOs reais.
  O benchmark do fabricante (Observatório Maritaca) é de tarefas gerais.
- **Memória mínima**: só o 0,25 foi medido. O piso estimado é menor (pesos
  8 GiB + ~8 GiB de KV para um pedido de contexto cheio), mas não foi testado.
- **Versão do vLLM**: validado na 0.22.1 (a fixada) com Python 3.14; a imagem
  usa Python 3.11. Uma atualização do vLLM pode tornar o `max_num_seqs`
  desnecessário.
- **Primeira execução** baixa ~9 GB e leva ~4 min para ficar pronta. Um cache
  de compilação persistente, como o do `ocr-worker` (ADR-012), cortaria cerca
  de 40 s por reinício (a compilação), mas não os ~80 s de perfil de memória.
  Não foi configurado para o `llm-worker`.
- **Não verificado**: a construção da imagem Docker; o Python 3.11 da imagem
  (o teste usou 3.14); e o comportamento com BOs reais.
