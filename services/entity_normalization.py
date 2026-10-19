"""The one copy of the entity-normalization rules for linkage.

Before this, the exact same constants and two normalization functions lived
twice — in scripts/backfill_linkage_scores.py and in
workers/linkage_scoring_worker.py — with a comment telling maintainers to keep
them in sync. They disagree silently when they drift: the script decides which
BO pairs exist from a normalized value, the scorer then compares the same BOs
by normalized value, so a difference means pairs get generated for values the
scorer sees as unrelated. No error, just missing links.

The rules mirror Maynara_cluster_v0/motor_linkage.py — they define the feature
space the linkage classifier (artifacts/bo_linkage_classifier_v1.joblib) was
trained on. Do not change without changing that reference too.
"""
import re
import unicodedata

ENTIDADES_ID = {
    "PESSOA_NOME", "DOCUMENTO", "PLACA", "BO_ID",
    "EMAIL", "TELEFONE", "INSTAGRAM", "URL",
    "CONTA_BANCARIA", "CHAVE_PIX", "OBJETO_VEICULO",
}
TIPOS_PESSOA = {"PESSOA_NOME"}
TIPOS_IDENTIFICADOR_DURO = {
    "DOCUMENTO", "PLACA", "BO_ID", "TELEFONE",
    "CONTA_BANCARIA", "AGENCIA_BANCARIA", "CHAVE_PIX",
}


def normalizar_nome(nome: str) -> str:
    nome = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode()
    return " ".join(nome.upper().split())


def normalizar_valor_entidade(texto: str, tipo: str) -> str:
    t = tipo.upper()
    if t in TIPOS_PESSOA:
        return normalizar_nome(texto)
    if t == "EMAIL":
        return texto.strip().lower()
    if t == "URL":
        return texto.strip().lower().rstrip("/")
    if t == "INSTAGRAM":
        return texto.strip().lower().lstrip("@")
    if t in TIPOS_IDENTIFICADOR_DURO:
        return re.sub(r"[^A-Z0-9]", "", texto.upper())
    txt = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    txt = re.sub(r"[^\w\s]", " ", txt.lower())
    return " ".join(txt.split())
