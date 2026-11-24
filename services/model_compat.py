"""Startup checks: do the models and artifacts fit each other?

The linkage classifier was calibrated on one encoder, one feature layout and
one file. If any of them changes, every score is wrong and nothing raises. These
checks turn that silent failure into a refusal to start (ADR-014).

The facts come from artifacts/models.lock.json. Set ALLOW_INCOMPATIBLE_MODELS
to run an experiment anyway; the worker then logs a warning instead.
"""
import logging
from pathlib import Path

from services.model_manifest import load_lock, sha256_file

logger = logging.getLogger(__name__)


class ModelCompatibilityError(RuntimeError):
    """The configured models do not fit the artifacts they depend on."""


def _fail(cfg, message: str) -> None:
    if cfg.allow_incompatible_models:
        logger.warning("Incompatible models allowed by ALLOW_INCOMPATIBLE_MODELS: %s", message)
        return
    raise ModelCompatibilityError(message)


def _classifier_facts(cfg) -> dict:
    return load_lock(cfg.model_lock_path).artifact(Path(cfg.linkage_classifier_path).name)


def check_encoder_matches_classifier(cfg) -> None:
    """The encoder must be the one the linkage classifier was calibrated with.

    Cheap and needs no model load, so both the embedding worker and the linkage
    worker call it when they start.
    """
    facts = _classifier_facts(cfg)
    expected = facts.get("encoder")
    if expected and cfg.embedding_model_name != expected:
        _fail(
            cfg,
            f"embedding_model_name is {cfg.embedding_model_name!r}, but the linkage "
            f"classifier was calibrated with {expected!r}. A different encoder gives "
            "a different vector space and silently invalidates every score. Retrain "
            "the classifier first, or set ALLOW_INCOMPATIBLE_MODELS=true to experiment.",
        )


def check_classifier_file(cfg) -> None:
    """The classifier file must be the locked one.

    A file that is not there is skipped: loading it fails on its own, and a
    developer machine has no /app/artifacts.
    """
    facts = _classifier_facts(cfg)
    expected = facts.get("sha256")
    if not expected:
        logger.warning("No locked hash for %s: its identity is not checked", cfg.linkage_classifier_path)
        return
    actual = sha256_file(cfg.linkage_classifier_path)
    if actual != "missing" and actual != expected:
        _fail(
            cfg,
            f"{cfg.linkage_classifier_path} has SHA-256 {actual[:12]}, but "
            f"artifacts/models.lock.json locks {expected[:12]}. Update the lock "
            "(scripts/update_model_lock.py) if the change is intended.",
        )


def check_loaded_classifier(cfg, classifier, feature_names) -> None:
    """The loaded classifier must expect exactly the features the worker builds."""
    expected = len(feature_names)
    actual = getattr(classifier, "n_features_in_", None)
    if actual != expected:
        _fail(
            cfg,
            f"The classifier expects {actual} features, but the worker builds "
            f"{expected} ({', '.join(feature_names)}).",
        )
