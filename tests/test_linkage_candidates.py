"""pares_por_grupo: the cap that keeps one generic entity value from becoming
O(k²) candidate pairs. See services/linkage_candidates.py."""
from services.linkage_candidates import pares_por_grupo


def test_expands_small_group_into_all_pairs():
    """Arrange: one group with three BOs, well under the cap.
    Act: expand it into candidate pairs.
    Assert: all three unordered pairs are emitted and nothing is skipped.
    """
    pares, pulados = pares_por_grupo({("PLACA", "ABC1234"): {1, 2, 3}}, max_group_size=100)

    assert pares == {(1, 2), (1, 3), (2, 3)}
    assert pulados == 0


def test_singleton_group_produces_nothing():
    """Arrange: one group containing a single BO.
    Act: expand it into candidate pairs.
    Assert: no pair is possible, so nothing is emitted or skipped.
    """
    pares, pulados = pares_por_grupo({("PLACA", "ABC1234"): {7}}, max_group_size=100)

    assert pares == set()
    assert pulados == 0


def test_group_over_the_cap_is_skipped_not_expanded():
    """Arrange: a generic entity shared by 1,000 BOs, over the 100 cap.
    Act: expand it into candidate pairs.
    Assert: the group is skipped and counted, avoiding ~499,500 pairs.
    """
    generico = set(range(1000))
    pares, pulados = pares_por_grupo(
        {("OBJETO_VEICULO", "motocicleta"): generico}, max_group_size=100
    )

    assert pares == set()  # 499.500 pairs avoided
    assert pulados == 1


def test_group_at_the_cap_is_expanded():
    """Arrange: a group with exactly 100 BOs, the configured cap.
    Act: expand it into candidate pairs.
    Assert: all 4,950 pairs are emitted and nothing is skipped.
    """
    no_limite = set(range(100))
    pares, pulados = pares_por_grupo({("PLACA", "ABC1234"): no_limite}, max_group_size=100)

    assert len(pares) == 100 * 99 // 2
    assert pulados == 0
