#!/usr/bin/env python3
"""Update artifacts/models.lock.json on purpose.

Dry run by default: it prints what would change and writes nothing. A model
upgrade is then one visible diff in a pull request, not whatever "latest" was
the day someone downloaded the model.

    python scripts/update_model_lock.py                       # show the changes
    python scripts/update_model_lock.py --write               # write them
    python scripts/update_model_lock.py --model maritaca-ai/S5-nano --write   # only this model

It asks Hugging Face for the newest commit of each pinned model (public data
only) and rehashes the local artifacts.
"""
import argparse
import copy
import datetime
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from services.model_manifest import sha256_file  # noqa: E402


def latest_revision(model: str) -> str:
    """The newest commit of `model` on the default branch."""
    url = f"https://huggingface.co/api/models/{model}"
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)["sha"]


def update_lock(
    lock: dict,
    artifacts_dir: Path,
    only: set[str] | None = None,
    fetch=None,
    today: str | None = None,
) -> tuple[dict, list[str]]:
    """Return the updated lock and a readable list of the changes.

    `fetch(model)` returns the newest commit. It defaults to latest_revision,
    looked up at call time so a test can replace it.
    """
    fetch = fetch or latest_revision
    today = today or datetime.date.today().isoformat()
    updated = copy.deepcopy(lock)
    changes: list[str] = []

    for model, entry in updated.get("huggingface", {}).items():
        if only and model not in only:
            continue
        newest = fetch(model)
        if newest != entry["revision"]:
            changes.append(f"{model}: {entry['revision'][:12]} -> {newest[:12]}")
            entry["revision"], entry["locked_at"] = newest, today

    for name, facts in updated.get("artifacts", {}).items():
        if only:
            continue
        digest = sha256_file(artifacts_dir / name)
        if digest != "missing" and digest != facts.get("sha256"):
            changes.append(f"artifacts/{name}: {str(facts.get('sha256'))[:12]} -> {digest[:12]}")
            facts["sha256"] = digest

    return updated, changes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="write the lock (default: dry run)")
    parser.add_argument("--model", action="append", help="update only this Hugging Face model (repeatable)")
    args = parser.parse_args(argv)

    lock_path = REPO / "artifacts/models.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    updated, changes = update_lock(
        lock, REPO / "artifacts", only=set(args.model) if args.model else None
    )

    if not changes:
        print("The lock is up to date.")
        return 0
    print("Changes:")
    for line in changes:
        print(f"  {line}")
    if args.write:
        lock_path.write_text(json.dumps(updated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {lock_path.relative_to(REPO)}. Review the diff, then test the new models.")
    else:
        print("Dry run: nothing written. Add --write to apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
