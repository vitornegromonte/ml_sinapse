# ADR-014: Identidade de modelo, linhagem das saídas e portões de CI

**Status:** ✅ Aceito<br>
**Data:** 2026-11-24

## Contexto

Em uma semana trocamos o modelo de OCR (Dharma) e o de resumo (S5-nano). Depois
da troca, nada nas saídas dizia qual modelo as escreveu: a mensagem de
resultado tinha só `document_id`, `text` e `text_length`. A revisão do projeto
(`ML_review.md`, F4 e F8) e o código mostravam as mesmas falhas:

- **Versão fixa no código.** `MODEL_VERSION` do worker de embedding era uma
  constante, mesmo com `EMBEDDING_MODEL_NAME` configurável. Trocar o encoder
  gravava a versão errada no `gold.model_registry`.
- **Metadado que não bate com o artefato.** O worker de linkage registrava
  `C=1000.0`, e o `.joblib` tem `C=10.0`.
- **Revisões não fixadas.** O vLLM e o sentence-transformers baixavam o que
  estivesse no branch padrão. O S5-nano foi baixado assim.
- **Nenhuma verificação na partida.** O classificador de linkage foi calibrado
  com um encoder, um layout de features e um arquivo. Mudar qualquer um deixa
  todo score errado, sem erro.
- **CI só com testes.** Nenhum portão para o lock, para o lint ou para a
  cobertura.

## Decisão

1. **Lock de modelos** em `artifacts/models.lock.json`, versionado no git: o
   commit (40 hex) de cada modelo do Hugging Face e, para cada artefato local,
   o SHA-256 e fatos (encoder de calibração, número de features, versão do
   scikit-learn do treino). `scripts/update_model_lock.py` o atualiza de
   propósito (simulação por padrão); `scripts/check_model_lock.py` o valida.
2. **Manifesto por estágio** (`services/model_manifest.py`): tudo o que muda o
   que o estágio escreve (modelo e revisão, hash dos prompts, amostragem, DPI,
   hash do classificador, limiar...). O `manifest_id` é o SHA-256 de 12
   caracteres disso. Ajustes de memória (fração da GPU, `max_num_seqs`) ficam
   de fora, pois não mudam o texto. O manifesto do linkage inclui o do encoder.
3. **A revisão fixada chega aos carregadores**: `revision` e
   `tokenizer_revision` no vLLM (OCR e LLM), `revision` no
   `SentenceTransformer` e no tokenizer do LLM.
4. **Linhagem nas saídas**, só com campos novos (nada é renomeado):
   - mensagens de `ocr:tasks:results`, `llm:tasks` e `scoring:tasks:results`
     ganham `manifest_id` e `model`;
   - objetos do bronze ganham esses dois campos como metadado (`x-amz-meta-*`);
   - cada worker grava o documento do manifesto em
     `ml-results/manifests/<manifest_id>`, uma vez por processo, para que um id
     possa ser consultado;
   - `metricas` de `gold.bo_relacionado` ganha `manifest_id`;
   - os hiperparâmetros do `gold.model_registry` saem do classificador carregado
     (não são digitados), e a versão do encoder vem da configuração.
5. **Verificações na partida** (`services/model_compat.py`): o worker de
   embedding e o de linkage recusam iniciar com um encoder diferente do da
   calibração; o de linkage recusa um `.joblib` cujo hash difere do lock e um
   classificador com número de features diferente. `ALLOW_INCOMPATIBLE_MODELS`
   transforma a recusa em aviso, para experimentos.
6. **Registro no MLflow** (`scripts/log_manifests_to_mlflow.py`): um run por
   manifesto, com os parâmetros e o JSON como artefato, idempotente. Métricas de
   avaliação poderão ser gravadas no mesmo run.
7. **CI**: `ruff check` (erros de sintaxe e pyflakes, `pyproject.toml`),
   `check_model_lock.py`, teste com cobertura mínima de 88% (medido: 91%), e uma
   verificação semanal e manual do lock contra o Hugging Face
   (`model-lock-online.yml`).

## Consequências

- **Positivas:** toda saída nova diz de que receita veio; trocar um modelo, um
  prompt ou o classificador muda o `manifest_id` e aparece em um diff do lock;
  um arquivo trocado ou um encoder incompatível derruba o contêiner na partida
  e não os scores.
- **Campos novos nos fluxos.** São só adições, mas o consumidor do backend
  precisa ignorar campos que não conhece. **Isso não foi verificado.**
- **Linhas antigas do `gold.model_registry` não mudam.** `get_or_create_model_id`
  só cria; a linha do linkage já gravada continua com `C=1000.0`. Corrigir é uma
  migração no backend.
- **Sem mudança de esquema** do Postgres, e o filtro por `model_id` nas
  consultas (F4) continua pendente: precisa de um banco para testar.
- **O `manifest_id` depende do ambiente** em dois pontos: o arquivo do
  classificador (`missing` onde `/app/artifacts` não existe) e a versão do
  sentence-transformers. Compare ids entre máquinas só em produção.
- **Os modelos locais** (um caminho em `LLM_MODEL_PATH`) não têm revisão: o
  manifesto os marca como `(unpinned)`.
- **Não coberto:** avaliação de qualidade, métricas de operação, tag e envio da
  imagem Docker, e as dependências do `requirements.txt` (a maioria sem versão
  fixa, por isso o `pip-audit` não foi adicionado: ele recusa requisitos sem
  versão).
