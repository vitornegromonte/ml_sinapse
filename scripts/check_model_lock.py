#!/usr/bin/env python3
"""Check artifacts/models.lock.json. Exit code 1 if anything is wrong.

CI runs this on every push. It catches the changes that would otherwise reach
production unseen: a default model with no pin, a pin that is not a commit
hash, an artifact whose content no longer matches its recorded hash, an
artifact nobody locked, and an encoder that no longer matches the classifier.

    python scripts/check_model_lock.py            # offline, fast
    python scripts/check_model_lock.py --online   # also ask Hugging Face that each pinned commit exists
"""
import argparse
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from config import Settings  # noqa: E402
from services.model_manifest import load_lock, sha256_file  # noqa: E402

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_LOCKED_SUFFIXES = {".joblib", ".json"}


def revision_exists(model: str, revision: str) -> bool:
    """Ask the Hugging Face API whether `revision` exists for `model`. Public data only."""
    url = f"https://huggingface.co/api/models/{model}/revision/{revision}"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.status == 200
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise


def check_lock(
    lock_path: Path,
    artifacts_dir: Path,
    cfg: Settings,
    feature_count: int | None = None,
    online: bool = False,
    exists=None,
) -> list[str]:
    """Return the problems found, as readable lines. An empty list means OK.

    `exists(model, revision)` defaults to revision_exists, looked up at call time.
    """
    exists = exists or revision_exists
    try:
        lock = load_lock(str(lock_path))
    except ValueError as exc:
        return [str(exc)]
    if not lock_path.is_file():
        return [f"{lock_path} does not exist"]

    problems: list[str] = []

    # 1. Every model the workers load by default is pinned.
    defaults = {
        "ocr_model_name": cfg.ocr_model_name,
        "llm_model_path": cfg.llm_model_path,
        "embedding_model_name": cfg.embedding_model_name,
    }
    for setting, model in defaults.items():
        if lock.revision(model) is None:
            problems.append(f"{model} (the default {setting}) is not pinned in the lock")

    # 2. Every pin is a full commit hash, not a branch name like "main".
    for model, revision in lock.huggingface.items():
        if not _COMMIT.match(revision):
            problems.append(f"{model}: revision {revision!r} is not a 40-character commit hash")

    # 3. Every locked artifact exists and has the recorded content.
    for name, facts in lock.artifacts.items():
        actual = sha256_file(artifacts_dir / name)
        if actual == "missing":
            problems.append(f"artifacts/{name} is in the lock but the file is missing")
        elif facts.get("sha256") != actual:
            problems.append(
                f"artifacts/{name} changed: SHA-256 is {actual[:12]}, the lock has "
                f"{str(facts.get('sha256'))[:12]} (run scripts/update_model_lock.py if intended)"
            )

    # 4. Every artifact file is locked: no model enters the image unseen.
    for file in sorted(artifacts_dir.iterdir()) if artifacts_dir.is_dir() else []:
        if file.suffix in _LOCKED_SUFFIXES and file.name != lock_path.name and file.name not in lock.artifacts:
            problems.append(f"artifacts/{file.name} is not in the lock")

    # 5. The classifier fits the default encoder and the worker's features.
    for name, facts in lock.artifacts.items():
        encoder = facts.get("encoder")
        if encoder and encoder != cfg.embedding_model_name:
            problems.append(
                f"artifacts/{name} was calibrated with {encoder!r}, but the default "
                f"embedding_model_name is {cfg.embedding_model_name!r}"
            )
        expected = facts.get("n_features")
        if expected is not None and feature_count is not None and expected != feature_count:
            problems.append(
                f"artifacts/{name} expects {expected} features, the worker builds {feature_count}"
            )

    # 6. Optional: the pinned commits exist on the Hub.
    if online:
        for model, revision in lock.huggingface.items():
            if _COMMIT.match(revision) and not exists(model, revision):
                problems.append(f"{model}@{revision[:12]} does not exist on Hugging Face")

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--online", action="store_true", help="also verify the pinned commits on Hugging Face")
    args = parser.parse_args(argv)

    from workers.linkage_scoring_worker import FEATURE_NAMES

    cfg = Settings(_env_file=None)
    lock_path = REPO / "artifacts/models.lock.json"
    problems = check_lock(lock_path, REPO / "artifacts", cfg, len(FEATURE_NAMES), online=args.online)

    if problems:
        print(f"{len(problems)} problem(s) in {lock_path.relative_to(REPO)}:")
        for line in problems:
            print(f"  - {line}")
        return 1
    print(f"{lock_path.relative_to(REPO)}: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
