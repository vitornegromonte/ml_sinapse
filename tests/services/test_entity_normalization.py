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
    """Arrange: a lowercase, accented person name.
    Act: normalize it as a name.
    Assert: accents are stripped and the result is uppercase.
    """
    assert normalizar_nome("joão da silva") == "JOAO DA SILVA"


def test_identificador_duro_keeps_only_alnum():
    """Arrange: a plate value with lowercase letters and a hyphen.
    Act: normalize it as a hard identifier.
    Assert: only alphanumerics survive, uppercased.
    """
    assert normalizar_valor_entidade("abc-1234", "PLACA") == "ABC1234"


def test_email_url_instagram():
    """Arrange: email, URL and Instagram values with stray whitespace/case.
    Act: normalize each by its entity type.
    Assert: each is trimmed and lowercased.
    """
    assert normalizar_valor_entidade(" A@B.COM ", "EMAIL") == "a@b.com"
    assert normalizar_valor_entidade("HTTPS://X.com/ ", "URL") == "https://x.com"
    assert normalizar_valor_entidade(" @Fulano ", "INSTAGRAM") == "fulano"


def test_fallback_strips_punctuation_and_accents():
    """Arrange: a free-text address with accents and punctuation.
    Act: normalize it with the fallback rule.
    Assert: accents and punctuation are removed, whitespace collapsed.
    """
    assert normalizar_valor_entidade("Rua São João, 42!", "LOCAL_AREA") == "rua sao joao 42"


def test_entidades_id_is_the_strong_identifier_set():
    """Arrange: the shared strong-identifier constant.
    Act: compare it with the expected set.
    Assert: exactly the 11 identifier types are included — adding or removing
    one changes which entity overlaps produce candidate pairs.
    """
    assert ENTIDADES_ID == {
        "PESSOA_NOME", "DOCUMENTO", "PLACA", "BO_ID",
        "EMAIL", "TELEFONE", "INSTAGRAM", "URL",
        "CONTA_BANCARIA", "CHAVE_PIX", "OBJETO_VEICULO",
    }
