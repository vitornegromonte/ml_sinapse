"""Maturity score for a BO (boletim de ocorrência) by ato infracional.

Ported from the `bos_scoring/` prototype. Pure code: no Redis, no DB, no
logging, no I/O beyond reading the weights file — the worker is only
transport. The split mirrors the prototype on purpose:

  * weights are data, one set per ato infracional, in
    `artifacts/bo_scoring_weights_v1.json` (every nature's weights sum to 1.0;
    `load_weights` refuses the file otherwise);
  * penalties are rules, keyed by nature in code (PENALTIES). Only homicídio
    and furto have one today; every other nature gets the neutral 1.0.

A feature the weights do not mention contributes nothing, so the payload can
carry extra flags without breaking the score. A nature with no weights is a
configuration error, not a data state: `score` raises.
"""

import json
from dataclasses import dataclass
from pathlib import Path


class UnknownNature(ValueError):
    """No weights configured for this ato infracional."""


def _no_penalty(features: dict[str, bool]) -> float:
    return 1.0


def homicide_penalty(features: dict[str, bool]) -> float:
    """Homicídio without an identified victim does not mature: 10%."""
    return 1.0 if features.get("victim", False) else 0.1


def theft_penalty(features: dict[str, bool]) -> float:
    """Furto without the stolen object AND an address keeps 30%."""
    return (
        1.0 if features.get("stolen_items") and features.get("address") else 0.3
    )


PENALTIES = {
    "HOMICIDIO": homicide_penalty,
    "FURTO": theft_penalty,
}


@dataclass(frozen=True)
class ScoringWeights:
    version: str
    by_nature: dict[str, dict[str, float]]

    def score(self, nature: str, features: dict[str, bool]) -> float:
        weights = self.by_nature.get(nature)
        if weights is None:
            raise UnknownNature(
                f"no scoring weights for ato_infracional '{nature}'"
            )
        linear = sum(
            weights.get(name, 0.0) * float(present)
            for name, present in features.items()
        )
        return linear * PENALTIES.get(nature, _no_penalty)(features)


def load_weights(path: str | Path) -> ScoringWeights:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    by_nature = {
        nature: {name: float(weight) for name, weight in weights.items()}
        for nature, weights in raw["natures"].items()
    }
    for nature, weights in by_nature.items():
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"scoring weights for '{nature}' sum to {total}, expected 1.0"
            )
    return ScoringWeights(version=str(raw.get("version", "")), by_nature=by_nature)
