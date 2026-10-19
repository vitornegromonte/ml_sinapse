"""Pins the shared normalization rules.

The generator (scripts/backfill_linkage_scores.py) and the scorer
(workers/linkage_scoring_worker.py) both read these; before B8 they were
duplicated verbatim and drifting apart was silent — pairs would be generated
for values the scorer then saw as unrelated. A change in behaviour should fail
here first.
"""
from services.entity_normalization import (
    ENTIDADES_ID,
    normalizar_nome,
    normalizar_valor_entidade,
)


def test_nome_is_accent_stripped_and_upper():
    assert normalizar_nome("joão da silva") == "JOAO DA SILVA"


def test_identificador_duro_keeps_only_alnum():
    assert normalizar_valor_entidade("abc-1234", "PLACA") == "ABC1234"


def test_email_url_instagram():
    assert normalizar_valor_entidade(" A@B.COM ", "EMAIL") == "a@b.com"
    assert normalizar_valor_entidade("HTTPS://X.com/ ", "URL") == "https://x.com"
    assert normalizar_valor_entidade(" @Fulano ", "INSTAGRAM") == "fulano"


def test_fallback_strips_punctuation_and_accents():
    assert normalizar_valor_entidade("Rua São João, 42!", "LOCAL_AREA") == "rua sao joao 42"


def test_entidades_id_is_the_strong_identifier_set():
    assert ENTIDADES_ID == {
        "PESSOA_NOME", "DOCUMENTO", "PLACA", "BO_ID",
        "EMAIL", "TELEFONE", "INSTAGRAM", "URL",
        "CONTA_BANCARIA", "CHAVE_PIX", "OBJETO_VEICULO",
    }
