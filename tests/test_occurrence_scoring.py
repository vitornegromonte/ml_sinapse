"""Port of the `bos_scoring/` prototype: same rules, weights moved to data.

The 180-case mock is kept as a fixture so a nature that exists in real data
but not in the weights file is caught here rather than in a DLQ at runtime.
"""
import json
from pathlib import Path

import pytest

from services.occurrence_scoring import UnknownNature, load_weights

WEIGHTS = Path("artifacts/bo_scoring_weights_v1.json")
FIXTURE = Path("tests/fixtures/mock_atos_infracionais.json")


@pytest.fixture(scope="module")
def weights():
    return load_weights(WEIGHTS)


@pytest.fixture(scope="module")
def cases():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_every_nature_in_the_data_has_weights(weights, cases):
    assert {case["ato_infracional"] for case in cases} <= set(weights.by_nature)


def test_weights_file_is_versioned(weights):
    assert weights.version == "v1"


def test_homicide_complete_scores_one(weights):
    features = {"victim": True, "suspect": True, "address": True, "witnesses": True}

    assert weights.score("HOMICIDIO", features) == pytest.approx(1.0)


def test_homicide_without_victim_keeps_a_tenth(weights):
    features = {"victim": False, "suspect": True, "address": True, "witnesses": True}

    assert weights.score("HOMICIDIO", features) == pytest.approx(0.5 * 0.1)


def test_theft_needs_both_the_object_and_an_address(weights):
    assert weights.score(
        "FURTO", {"stolen_items": True, "address": True}
    ) == pytest.approx(0.6)
    assert weights.score("FURTO", {"stolen_items": True}) == pytest.approx(0.4 * 0.3)


def test_nature_without_a_penalty_rule_is_linear(weights):
    # AMEACA: victim 0.4 + witnesses 0.1, no specific penalty.
    assert weights.score(
        "AMEACA", {"victim": True, "witnesses": True}
    ) == pytest.approx(0.5)


def test_unknown_feature_contributes_nothing(weights):
    assert weights.score(
        "AMEACA", {"victim": True, "not_a_real_feature": True}
    ) == pytest.approx(0.4)


def test_unknown_nature_raises(weights):
    with pytest.raises(UnknownNature, match="ATO_QUE_NAO_EXISTE"):
        weights.score("ATO_QUE_NAO_EXISTE", {"victim": True})


def test_every_case_in_the_mock_scores_between_zero_and_one(weights, cases):
    for case in cases:
        score = weights.score(case["ato_infracional"], case["features"])
        assert 0.0 <= score <= 1.0


def test_weights_that_do_not_sum_to_one_are_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"version": "x", "natures": {"X": {"a": 0.5}}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sum to 0.5"):
        load_weights(path)


def test_missing_weights_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_weights(tmp_path / "nope.json")
