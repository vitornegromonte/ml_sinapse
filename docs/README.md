# ML Service — Architecture Decision Records

Este diretório contém os ADRs (_Architecture Decision Records_) do subsistema de ML.

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

Os ADRs mantêm o corpo original; a seção **Atualização** no fim de cada um
registra o que mudou depois.

## Outros documentos

- [Testing](./testing.md) — organização da suíte e a convenção Arrange, Act, Assert.
