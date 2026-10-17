# ADR-006: Pesos de modelo locais

**Status:** ✅ Aceito  
**Data:** 2026-06-02  

## Contexto

Os workers precisam acessar os pesos dos modelos. As opções são: baixar do HuggingFace a cada execução, incluir na imagem Docker, ou montar via volume.

## Decisão

Usar **pesos locais em disco** com cache via Docker volume, com fallback para download automático no primeiro uso:

- Diretório base: `/model-cache` (Docker volume configurável via `model_base_path`)
- Modelos baixados na primeira execução e cacheados
- Workers carregam do path local (sem depender de internet após o cache)
- Alternativa futura: MLflow Registry carregando do bucket `models` no MinIO

```
volume model_cache:/model-cache
  ├── huggingface/   (OCR, tokenizer)
  ├── spacy/         (NER mock)
  └── bertimbau/     (tokenizer)
```

## Consequências

- **Positivas:** Imagem Docker leve (modelos não incluídos na build); cache persistente entre execuções; fácil troca de versão (só substituir o volume); compatível com MLflow Registry futuro.
- **Negativas:** Primeira execução demora (download); volume ocupado em disco (~2-5GB por modelo); workers precisam de permissão de escrita no volume.
