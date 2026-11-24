# ML Service — Architecture Decision Records

Este diretório contém os ADRs (_Architecture Decision Records_) do subsistema de ML.
Para a visão geral do sistema, veja [architecture.md](./.claude/architecture.md).

| ADR | Título | Status |
|---|---|---|
| [ADR-001](./adr-001-service-structure.md) | Estrutura do serviço de ML | ✅ Aceito (histórico) |
| [ADR-002](./adr-002-redis-streams-workers.md) | Workers assíncronos com Redis Streams | ✅ Aceito (atualizado) |
| [ADR-003](./adr-003-multiprocess-workers.md) | Isolamento de workers via multiprocessing | ✅ Aceito (atualizado) |
| [ADR-004](./adr-004-multimodal-ocr.md) | OCR multimodal via VLM | ✅ Aceito (atualizado) |
| [ADR-005](./adr-005-nermodel-spacy-mock.md) | NER mock com spaCy (provisório) | ✅ Aceito |
| [ADR-006](./adr-006-local-weights.md) | Pesos de modelo locais | ✅ Aceito (atualizado) |
| [ADR-007](./adr-007-minio-lakehouse.md) | Data lake Bronze/Prata/Ouro com MinIO | ✅ Aceito |
| [ADR-008](./adr-008-mlflow-registry.md) | MLflow para experimentos e registry | ✅ Aceito |
| [ADR-009](./adr-009-model-wrappers.md) | Wrappers de modelo modulares (BaseModelWrapper) | 🟡 Parcialmente substituído |
| [ADR-010](./adr-010-bentoml-serving.md) | Serving com BentoML | ⛔ Substituído |
| [ADR-011](./adr-011-permanent-errors-and-model-timeout.md) | Falhas permanentes e timeout de modelo | ✅ Aceito |
| [ADR-012](./adr-012-ocr-throughput-and-resume.md) | Velocidade do OCR e retomada após falha | ✅ Aceito |
| [ADR-013](./adr-013-llm-s5-nano.md) | LLM de resumo: maritaca-ai/S5-nano | ✅ Aceito |
| [ADR-014](./adr-014-model-identity-and-lineage.md) | Identidade de modelo, linhagem das saídas e portões de CI | ✅ Aceito |

Os ADRs mantêm o corpo original; a seção **Atualização** no fim de cada um
registra o que mudou depois.

## Outros documentos

- [Testing](./.claude/testing.md) — organização da suíte e a convenção Arrange, Act, Assert.
