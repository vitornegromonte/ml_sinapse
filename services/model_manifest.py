"""Model identity: what exactly produced an output.

A **manifest** is everything that changes what a stage writes: the model and its
pinned revision, the prompts, the sampling, the artifact hashes. Its
**manifest_id** is a short hash of that, so two outputs with the same
manifest_id came from the same recipe, and a different id means the recipe
changed. Workers put the id on their outputs (see ADR-014).

The pins live in artifacts/models.lock.json, a file in git, so a model upgrade
is a reviewed change and not a side effect of whatever "latest" was at
download time.

Imports of the heavy modules (torch, the workers) are inside the functions: the
CI lock check and the tests must be able to import this module cheaply.
"""
import hashlib
import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import metadata
from pathlib import Path

logger = logging.getLogger(__name__)

LOCK_VERSION = 1
STAGES = ("ocr", "llm", "embedding", "linkage", "cluster", "scoring", "ner")
_ID_LENGTH = 12


def sha256_file(path: str | Path) -> str:
    """Hex SHA-256 of a file, read in chunks. "missing" if the file is absent."""
    file = Path(path)
    if not file.is_file():
        return "missing"
    digest = hashlib.sha256()
    with file.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(*parts: str) -> str:
    """Short hash of text parts, for prompts."""
    digest = hashlib.sha256("\n\x1f\n".join(parts).encode("utf-8"))
    return digest.hexdigest()[:_ID_LENGTH]


@dataclass(frozen=True)
class ModelLock:
    """The parsed lock file."""

    huggingface: dict[str, str] = field(default_factory=dict)  # model id -> commit
    artifacts: dict[str, dict] = field(default_factory=dict)  # file name -> facts

    def revision(self, model: str) -> str | None:
        """The pinned commit of a Hugging Face model, or None if it is not pinned."""
        return self.huggingface.get(model) or None

    def artifact(self, name: str) -> dict:
        return self.artifacts.get(name, {})


@lru_cache(maxsize=8)
def load_lock(path: str) -> ModelLock:
    """Read the lock file. A missing file gives an empty lock (nothing pinned).

    A file that exists but is not valid raises ValueError: a broken lock must
    stop the worker, not silently unpin every model.
    """
    file = Path(path)
    if not file.is_file():
        logger.warning("Model lock %s not found: no model is pinned", path)
        return ModelLock()
    try:
        raw = json.loads(file.read_text(encoding="utf-8"))
        if raw.get("lock_version") != LOCK_VERSION:
            raise ValueError(f"lock_version must be {LOCK_VERSION}")
        huggingface = {
            name: str(entry["revision"]) for name, entry in raw.get("huggingface", {}).items()
        }
        artifacts = {name: dict(entry) for name, entry in raw.get("artifacts", {}).items()}
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError, ValueError) as exc:
        raise ValueError(f"Invalid model lock {path}: {exc}") from exc
    return ModelLock(huggingface=huggingface, artifacts=artifacts)


def revision_for(model: str, cfg) -> str | None:
    """The pinned revision for `model`, for the loader to pass to Hugging Face.

    None means "not pinned": the loader fetches whatever the default branch
    holds. The manifest records that fact.
    """
    return load_lock(cfg.model_lock_path).revision(model)


@dataclass(frozen=True)
class StageManifest:
    stage: str
    model: str  # readable: "name@<12 hex of the commit>" or "name (unpinned)"
    identity: dict
    manifest_id: str

    def lineage(self) -> dict[str, str]:
        """The two fields that go on every output of this stage."""
        return {"manifest_id": self.manifest_id, "model": self.model}

    def to_json(self) -> str:
        document = {
            "manifest_id": self.manifest_id,
            "stage": self.stage,
            "model": self.model,
            "identity": self.identity,
        }
        return json.dumps(document, sort_keys=True, indent=2, default=str)


def _model_ref(model: str, revision: str | None) -> str:
    return f"{model}@{revision[:_ID_LENGTH]}" if revision else f"{model} (unpinned)"


def _make(
    stage: str, model: str, revision: str | None, identity: dict, ref: str | None = None
) -> StageManifest:
    """Build a manifest. `ref` replaces the readable label for stages whose pin
    is not a Hugging Face commit (an artifact hash, a version)."""
    body = {"stage": stage, "model": model, "revision": revision, **identity}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    manifest_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:_ID_LENGTH]
    return StageManifest(stage, ref or _model_ref(model, revision), body, manifest_id)


def _library_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def publish_manifest(storage, manifest: StageManifest) -> None:
    """Write the manifest document to bronze, so a manifest_id can be looked up.

    Best effort: a MinIO fault is logged and skipped, because the outputs carry
    the id anyway and the next call tries again. The path is
    ml-results/manifests/<manifest_id>.
    """
    try:
        storage.save(
            prefix="manifests",
            document_id=manifest.manifest_id,
            data=manifest.to_json().encode("utf-8"),
            content_type="application/json",
        )
    except Exception as exc:
        logger.warning("Could not publish manifest %s: %s", manifest.manifest_id, exc)


# ------------------------------------------------------------ stage builders


def _ocr(cfg) -> StageManifest:
    from models.ocr.registry import model_class_for

    model = cfg.ocr_model_uri or cfg.ocr_model_name
    revision = None if cfg.ocr_model_uri else revision_for(cfg.ocr_model_name, cfg)
    klass = None if cfg.ocr_model_uri else model_class_for(cfg.ocr_model_name)
    prompts = sha256_text(
        getattr(klass, "SYSTEM_PROMPT", ""), getattr(klass, "USER_PROMPT", "")
    )
    return _make("ocr", model, revision, {
        "prompts_sha": prompts,
        "repetition_penalty": getattr(klass, "REPETITION_PENALTY", None),
        "render_dpi": cfg.ocr_render_dpi,
        "max_tokens": cfg.ocr_max_tokens,
        "max_model_len": cfg.ocr_max_model_len,
        "dtype": cfg.ocr_dtype,
    })


def _llm(cfg) -> StageManifest:
    return _make("llm", cfg.llm_model_path, revision_for(cfg.llm_model_path, cfg), {
        "prompt_sha": sha256_text(cfg.llm_prompt_template),
        "sampling_params": cfg.llm_sampling_params or {},
        "chat_template_kwargs": cfg.llm_chat_template_kwargs or {},
        "dtype": cfg.llm_dtype,
        "max_model_len": cfg.llm_max_model_len,
    })


def _embedding(cfg) -> StageManifest:
    return _make("embedding", cfg.embedding_model_name,
                 revision_for(cfg.embedding_model_name, cfg), {
        "library": "sentence-transformers",
        "library_version": _library_version("sentence-transformers"),
    })


def _linkage(cfg) -> StageManifest:
    from workers.linkage_scoring_worker import FEATURE_NAMES, SCHEMA_ENTIDADES

    classifier_sha = sha256_file(cfg.linkage_classifier_path)
    return _make("linkage", "bo_linkage_classifier", classifier_sha, {
        "classifier_sha256": classifier_sha,
        # The score is only valid for the encoder it was calibrated with, so the
        # encoder's identity is part of the linkage identity.
        "encoder_manifest_id": _embedding(cfg).manifest_id,
        "feature_names": list(FEATURE_NAMES),
        "entity_schema_sha": sha256_text(*SCHEMA_ENTIDADES),
        "tau_dias": cfg.linkage_tau_dias,
        "limiar_descricao": cfg.linkage_limiar_descricao,
    }, ref=f"bo_linkage_classifier@{classifier_sha[:_ID_LENGTH]}")


def _cluster(cfg) -> StageManifest:
    return _make("cluster", "bo_linkage_cluster", None, {
        "algorithm": "connected-components",
        "score_threshold": cfg.cluster_score_threshold,
    }, ref=f"bo_linkage_cluster@threshold-{cfg.cluster_score_threshold}")


def _scoring(cfg) -> StageManifest:
    weights = Path(cfg.scoring_weights_path)
    version = ""
    if weights.is_file():
        version = str(json.loads(weights.read_text(encoding="utf-8")).get("version", ""))
    weights_sha = sha256_file(weights)
    return _make("scoring", "bo_maturity_score", weights_sha, {
        "weights_sha256": weights_sha,
        "weights_version": version,
    }, ref=f"bo_maturity_score@{version or weights_sha[:_ID_LENGTH]}")


def _ner(cfg) -> StageManifest:
    model = cfg.ner_model_uri or cfg.ner_model_minio_prefix
    return _make("ner", model, None, {"source": "mlflow" if cfg.ner_model_uri else "minio"},
                 ref=model)


_BUILDERS = {
    "ocr": _ocr, "llm": _llm, "embedding": _embedding, "linkage": _linkage,
    "cluster": _cluster, "scoring": _scoring, "ner": _ner,
}


def build_manifest(stage: str, cfg) -> StageManifest:
    """The manifest of `stage` under the settings `cfg`."""
    try:
        builder = _BUILDERS[stage]
    except KeyError:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {STAGES}") from None
    return builder(cfg)
