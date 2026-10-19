"""Candidate-pair generation for the linkage backfill.

Pure and DB-free so the one non-obvious rule here — a generic entity value
shared by thousands of BOs explodes into k²/2 candidate pairs — lives in one
place with a test. Used by scripts/backfill_linkage_scores.py.
"""


def pares_por_grupo(
    grupos: dict[tuple[str, str], set[int]],
    max_group_size: int,
) -> tuple[set[tuple[int, int]], int]:
    """Expand each (label, normalized value) group into unordered pairs.

    Returns (pares, grupos_pulados). A group bigger than `max_group_size` is
    treated as a generic value (OBJETO_VEICULO capturing "Motocicleta", say)
    rather than an identifier, and is skipped instead of expanded: that one
    group is what turns thousands of BOs into millions of pairs — a value
    shared by all 10.518 BOs of today would be ~55M pairs on its own.
    """
    pares: set[tuple[int, int]] = set()
    grupos_pulados = 0
    for bo_ids in grupos.values():
        if len(bo_ids) < 2:
            continue
        if len(bo_ids) > max_group_size:
            grupos_pulados += 1
            continue
        ordenados = sorted(bo_ids)
        for i, a in enumerate(ordenados):
            for b in ordenados[i + 1 :]:
                pares.add((a, b))
    return pares, grupos_pulados
