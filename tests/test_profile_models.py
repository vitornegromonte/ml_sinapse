"""O conselho de roteamento do profiler: é aritmética que decide a config do
compose, então vale um teste — o resto do script precisa de GPU."""
from scripts.profile_models import _resumo, _tabela


def _linha(modelo, **troca):
    base = {
        "modelo": modelo,
        "gpu": 0,
        "placa_mb": 24564,
        "contexto_mb": 400,
        "modelo_mb": 500,
        "pico_mb": 500,
        "torch_reservado_pico_mb": 450,
        "util": None,
        "carga_s": 1.0,
        "inferencia_s": 0.1,
    }
    return base | troca


def test_resumo_desconta_os_fixos_do_util_do_vllm():
    linhas = [
        _linha("embedding"),
        _linha("linkage"),
        _linha("ocr", util=0.9, pico_mb=22000),
    ]

    texto = _resumo(linhas)

    # 24564 - (500 + 500) = 23564 → 0.9593
    assert "--util <= 0.96" in texto
    assert "--util <= 0.48" in texto  # dois vLLM na mesma placa
    assert "embedding + linkage" in texto


def test_resumo_nao_arrisca_um_roteamento_com_so_um_modelo():
    assert _resumo([_linha("llm", util=0.9)]) == ""


def test_tabela_marca_os_fixos_sem_util_e_mostra_mb():
    tabela = _tabela([_linha("embedding", modelo_mb=520), _linha("llm", util=0.9)])

    linhas = tabela.splitlines()
    assert linhas[0].split()[0] == "modelo"
    assert "520" in linhas[1]
    assert linhas[1].rstrip().endswith("-")  # util não se aplica a modelo fixo
    assert "0.90" in linhas[2]
    assert tabela.endswith("(MB)\n")
