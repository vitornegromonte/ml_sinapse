# ADR-007: Data lake Bronze/Prata/Ouro com MinIO

**Status:** ✅ Aceito  
**Data:** 2026-06-02  

## Contexto

O sistema precisa armazenar documentos brutos (BOs, inquéritos), artefatos de MLflow e pesos de modelo versionados. Esses dados têm diferentes requisitos de acesso, retenção e política de segurança.

## Decisão

Implementar uma arquitetura de **data lake em três camadas** usando MinIO (S3-compatible) e PostgreSQL:

```
MinIO                              PostgreSQL
─────────────────                  ─────────────────
bronze/          [private]         prata/ (schemas)
├── bos/{ano}/                      ├── ocr_texts
│   ├── original.pdf                ├── ner_entities
│   └── anexos/                     ├── tokenized_texts
└── inqueritos/{id}/                └── embeddings
    └── documento.pdf
                                    ouro/ (schemas)
mlflow-artifacts/ [private]          ├── features_bo
└── experimentos/                    ├── similarity_matrix
                                     └── series
models/          [download]
└── ner/, ocr/, llm/               mlflow_db (tabelas MLflow)
```

**Políticas de acesso:**
| Bucket | Leitura | Escrita | Delete |
|---|---|---|---|
| `bronze` | Backend endpoint | Backend endpoint | ❌ Append-only |
| `mlflow-artifacts` | MLflow server | MLflow server | ❌ |
| `models` | **Público (download)** | MLflow (promote) | ❌ |

O `models` tem policy `download` para que workers possam baixar pesos sem chave de API, enquanto `bronze` e `mlflow-artifacts` são `private`.

## Consequências

- **Positivas:** Separação clara de responsabilidades por camada; políticas de acesso granulares por bucket; MinIO é S3-compatible (migração trivial para AWS S3 ou CloudFlare R2); MLflow usa o bucket `mlflow-artifacts` como artifact store nativo.
- **Negativas:** Mais um serviço para gerenciar (MinIO + mc-init); necessidade de criar buckets na inicialização; backup de volumes do MinIO.
- **Alternativas rejeitadas:** Sistema de arquivos local (não escala, sem política de acesso), AWS S3 direto (dependência externa para dev), HDFS (overhead desnecessário).
