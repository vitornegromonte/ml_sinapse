# ADR-005: NER mock com spaCy (provisório)

**Status:** ✅ Aceito  
**Data:** 2026-06-02  

## Contexto

O NER precisa extrair entidades de BOs (pessoas, placas, locais, órgãos, valores etc.). O modelo final será um BERTimbau fine-tuned em corpus de BOs, mas durante o desenvolvimento precisamos de um modelo funcional.

## Decisão

Usar **spaCy** com um modelo treinado localmente (`model-best/`) como mock durante o desenvolvimento. Este modelo:

- Foi treinado com `spacy train` a partir de dados de BOs
- Pipeline: `tok2vec` (MultiHashEmbed + MaxoutWindowEncoder) → `ner` (TransitionBasedParser)
- 11 labels: `ARTIGO_LEI`, `CARGO`, `CONTATO`, `DATA`, `DOCUMENTO`, `HORA`, `LOCAL`, `ORGAO`, `PESSOA`, `PLACA`, `VALOR`
- F1 global ≈ 0.45 (qualidade de desenvolvimento/mock)
- Carregado do path local `model-best/`

## Consequências

- **Positivas:** Mock funcional desde o início; pipeline spaCy é simples e rápido para prototipação; 11 labels alinhadas com o domínio policial.
- **Negativas:** F1 baixo (não usado em produção); transição futura para BERTimbau exigirá reescrita do worker; integração MLflow precisará ser testada na migração.
