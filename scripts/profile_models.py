#!/usr/bin/env python3
"""Mede a VRAM de cada worker de modelo — insumo para decidir o roteamento de GPU.

Uso:
    python scripts/profile_models.py --dry-run            # só mostra o plano
    python scripts/profile_models.py --model llm --gpu 1
    python scripts/profile_models.py --all --gpu 0        # um por processo
    python scripts/profile_models.py --all --gpu 0 --json

Cada modelo é medido em um processo próprio (`--all` reexecuta este script),
porque cada worker tem o próprio contexto CUDA (~0.3-0.6 GB) e porque o vLLM
não devolve a VRAM de forma confiável dentro do mesmo processo.

`--gpu` é aplicado pinando o filho com CUDA_VISIBLE_DEVICES, não passando o
índice ao modelo: o vLLM escolhe a placa só pela variável de ambiente.

`ocr` e `llm` rodam em vLLM: eles NÃO têm um tamanho intrínseco — reservam
`*_GPU_MEMORY_UTILIZATION` da placa na carga. O número medido é o custo do
`--util` usado; repita com o valor que pretende rodar. Como dividir uma GPU:
carregue primeiro os modelos de tamanho fixo (`embedding`, `linkage`) e use
--util ≤ (total - já ocupado) / total.

MlflowOcrModel fica de fora: exige um registry MLflow vivo, que este script
não tem como adivinhar.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings  # noqa: E402

MARCADOR = "__PERFIL__"
MODELOS = ["ocr", "llm", "embedding", "linkage"]
FIXOS = ("embedding", "linkage")  # não-vLLM: tamanho medido é o tamanho real


def _carregar(nome: str, gpu: int, util: float, max_len: int | None, cache_dir: str):
    if nome == "ocr":
        from models.ocr import create_model

        modelo = create_model(
            settings.ocr_model_name,
            cache_dir=cache_dir,
            gpu_device=gpu,
            gpu_memory_utilization=util,
            max_model_len=max_len,
        )
        modelo.load()
        return modelo

    if nome == "llm":
        from models.llm_model import LLMModel

        if not settings.llm_model_path:
            raise SystemExit("LLM_MODEL_PATH vazio — exporte o id do modelo (o compose já passa)")
        modelo = LLMModel(
            model_path=settings.llm_model_path,
            # "auto" = a única placa visível. Passar "cuda:0" faria o LLMModel
            # reescrever CUDA_VISIBLE_DEVICES e os subprocessos do vLLM
            # herdariam 0 (a placa física errada).
            device="auto",
            gpu_memory_utilization=util,
            max_model_len=max_len,
            sampling_params={"max_tokens": 32},
        )
        modelo.load()
        return modelo

    # embedding e linkage usam o mesmo encoder; são workers distintos, então
    # cada um paga o próprio contexto CUDA. O classificador do linkage é CPU.
    from models.embedding_model import SentenceTransformerEncoder

    encoder = SentenceTransformerEncoder(settings.embedding_model_name, device=f"cuda:{gpu}")
    encoder.load()
    return encoder


def _usar(nome: str, modelo) -> None:
    if nome == "ocr":
        from PIL import Image

        modelo.predict(Image.new("RGB", (1024, 1024), "white"), 32)
    elif nome == "llm":
        modelo.predict("Responda apenas: ok")
    else:
        modelo.encode(["descrição de teste para medir a alocação da GPU"])


def _medir(nome: str, gpu: int, util: float, max_len: int | None, cache_dir: str) -> dict:
    import torch

    # O processo já foi pinado por CUDA_VISIBLE_DEVICES, então a única placa
    # visível é a 0. Não adianta passar `--gpu` ao modelo: o vLLM ignora
    # device no construtor e usa a variável de ambiente.
    visivel = 0

    torch.zeros(1, device=f"cuda:{visivel}")  # força o contexto CUDA antes da baseline
    torch.cuda.synchronize(visivel)
    livre_antes, total = torch.cuda.mem_get_info(visivel)
    usado_antes = total - livre_antes
    torch.cuda.reset_peak_memory_stats(visivel)

    inicio = time.perf_counter()
    modelo = _carregar(nome, visivel, util, max_len, cache_dir)
    torch.cuda.synchronize(visivel)
    segundos_carga = time.perf_counter() - inicio
    livre_carregado, _ = torch.cuda.mem_get_info(visivel)

    inicio = time.perf_counter()
    _usar(nome, modelo)
    torch.cuda.synchronize(visivel)
    segundos_inferencia = time.perf_counter() - inicio
    livre_final, _ = torch.cuda.mem_get_info(visivel)

    mb = lambda bytes_: int(bytes_ // 2**20)  # noqa: E731
    return {
        "modelo": nome,
        "gpu": gpu,
        "placa_mb": mb(total),
        "contexto_mb": mb(usado_antes),
        "modelo_mb": mb(total - livre_carregado - usado_antes),
        "pico_mb": mb(total - livre_final - usado_antes),
        "torch_reservado_pico_mb": mb(torch.cuda.max_memory_reserved(visivel)),
        # util só significa algo para vLLM; nos fixos o tamanho é o tamanho.
        "util": None if nome in FIXOS else util,
        "carga_s": round(segundos_carga, 1),
        "inferencia_s": round(segundos_inferencia, 2),
    }


def _util_padrao(nome: str) -> float:
    if nome == "ocr":
        return settings.ocr_gpu_memory_utilization
    if nome == "llm":
        return settings.llm_gpu_memory_utilization
    return 1.0  # não se aplica: não é vLLM


def _tabela(linhas: list[dict]) -> str:
    saida = [
        f"{'modelo':<10} {'gpu':>3} {'placa':>7} {'contexto':>9} {'modelo':>7} "
        f"{'pico':>6} {'carga':>7} {'util':>5}"
    ]
    for r in linhas:
        util = "-" if r["util"] is None else f"{r['util']:.2f}"
        saida.append(
            f"{r['modelo']:<10} {r['gpu']:>3} {r['placa_mb']:>7} {r['contexto_mb']:>9} "
            f"{r['modelo_mb']:>7} {r['pico_mb']:>6} {r['carga_s']:>6}s {util:>5}"
        )
    return "\n".join([*saida, "(MB)"]) + "\n"


def _resumo(linhas: list[dict]) -> str:
    fixos = [r for r in linhas if r["modelo"] in FIXOS]
    vllm = [r["modelo"] for r in linhas if r["modelo"] not in FIXOS]
    if not fixos or not vllm:
        return ""
    placa = linhas[0]["placa_mb"]
    total_fixos = sum(r["pico_mb"] for r in fixos)
    livre = (placa - total_fixos) / placa
    return (
        f"\nFixos ({' + '.join(r['modelo'] for r in fixos)}), um processo cada: "
        f"{total_fixos} MB de {placa} MB.\n"
        f"Um modelo vLLM na mesma GPU deles: --util <= {livre:.2f}.\n"
        f"Dois vLLM na mesma GPU: --util <= {livre / 2:.2f} cada "
        "(evite: dois contextos e dois allocators na mesma placa).\n"
        "vLLM reserva util x placa na carga — carregue emb/linkage primeiro.\n"
    )


def _rodar_um(nome: str, gpu: int, util: float, max_len: int | None, cache_dir: str) -> dict:
    """Mede em um processo novo, para o contexto CUDA não ser compartilhado."""
    comando = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model", nome,
        "--gpu", str(gpu),
        "--util", str(util),
        "--cache-dir", cache_dir,
        "--emitir",
    ]
    if max_len:
        comando += ["--max-len", str(max_len)]
    # Pinar aqui, e não dentro do modelo: o vLLM escolhe a placa por
    # CUDA_VISIBLE_DEVICES, e GlmOcrModel não pina nada.
    ambiente = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    saida = subprocess.run(comando, capture_output=True, text=True, env=ambiente)
    for linha in saida.stdout.splitlines():
        if linha.startswith(MARCADOR):
            return json.loads(linha[len(MARCADOR):])
    erro = (saida.stderr or saida.stdout).strip().splitlines()
    raise SystemExit(f"{nome}: falhou — {erro[-1] if erro else 'sem saída'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODELOS)
    parser.add_argument("--all", action="store_true", help="mede todos, um processo por modelo")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--util", type=float, help="gpu_memory_utilization do vLLM (padrão: o de config)")
    parser.add_argument("--max-len", type=int)
    parser.add_argument("--cache-dir", default="/tmp/models")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--emitir", action="store_true", help=argparse.SUPPRESS)  # usado por --all
    args = parser.parse_args()

    if not args.all and not args.model and not args.dry_run:
        parser.error("use --model NOME ou --all")

    nomes = MODELOS if (args.all or not args.model) else [args.model]
    util = args.util

    if args.dry_run:
        for nome in nomes:
            print(f"{nome:<10} gpu={args.gpu} util={util if util is not None else _util_padrao(nome)} cache={args.cache_dir}")
        return

    if args.emitir:
        resultado = _medir(args.model, args.gpu, util if util is not None else _util_padrao(args.model), args.max_len, args.cache_dir)
        print(MARCADOR + json.dumps(resultado))
        return

    linhas, falhas = [], []
    for nome in nomes:
        try:
            linhas.append(_rodar_um(nome, args.gpu, util if util is not None else _util_padrao(nome), args.max_len, args.cache_dir))
        except SystemExit as exc:  # um modelo ausente não deve apagar os outros
            falhas.append(str(exc))

    if args.json:
        print(json.dumps(linhas, indent=2))
    elif linhas:
        print(_tabela(linhas))
        print(_resumo(linhas))
        for r in linhas:
            if r["modelo_mb"] <= 0:
                print(
                    f"aviso: {r['modelo']} não alocou nada na GPU {r['gpu']} — "
                    "suspeite de carga em outra placa",
                    file=sys.stderr,
                )
    for falha in falhas:
        print(falha, file=sys.stderr)
    if falhas and not linhas:
        sys.exit(1)


if __name__ == "__main__":
    main()
