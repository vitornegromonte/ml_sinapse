"""O conselho de roteamento do profiler: é aritmética que decide a config do
compose, então vale um teste — o resto do script precisa de GPU."""
from types import SimpleNamespace

import pytest

from scripts import profile_models
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


def test_rodar_um_pina_o_filho_na_gpu_pedida(monkeypatch):
    chamada = {}

    def falso(comando, **kwargs):
        chamada["env"] = kwargs["env"]
        return SimpleNamespace(stdout='__PERFIL__{"modelo": "ocr", "gpu": 1}', stderr="")

    monkeypatch.setattr(profile_models.subprocess, "run", falso)

    linha = profile_models._rodar_um("ocr", 1, 0.9, None, "/tmp/models")

    # Foi este detalhe que fez o primeiro run medir 0 MB: sem pinar o
    # processo, o GLM-OCR carrega na GPU 0 e a medição na 1 não vê nada.
    assert chamada["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert linha["gpu"] == 1


def test_rodar_um_mostra_o_erro_do_filho(monkeypatch):
    monkeypatch.setattr(
        profile_models.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout="", stderr="Traceback\nCUDA out of memory"),
    )

    with pytest.raises(SystemExit, match="CUDA out of memory"):
        profile_models._rodar_um("llm", 0, 0.9, None, "/tmp/models")
