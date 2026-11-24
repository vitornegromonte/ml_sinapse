#!/usr/bin/env python3
"""Record the model manifests in MLflow, one run per manifest.

The workers put a manifest_id on every output (ADR-014). This script makes each
id browsable: a run with the manifest's settings as parameters and its JSON as
an artifact. The same run can later receive evaluation metrics, so a recipe and
its measured quality live together.

It is idempotent: a manifest_id that already has a run is skipped.

    MLFLOW_TRACKING_URI=http://mlflow:5000 python scripts/log_manifests_to_mlflow.py
    python scripts/log_manifests_to_mlflow.py --tracking-uri sqlite:///mlflow.db --stage llm --stage ocr
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from config import Settings  # noqa: E402
from services.model_manifest import STAGES, build_manifest  # noqa: E402

DEFAULT_EXPERIMENT = "ml-sinapse/manifests"
_PARAM_LIMIT = 500  # MLflow's classic limit for a parameter value


def flatten(identity: dict, prefix: str = "") -> dict[str, str]:
    """Flatten nested dicts to dotted keys. Values become short strings."""
    flat: dict[str, str] = {}
    for key, value in identity.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, prefix=f"{name}."))
        else:
            text = value if isinstance(value, str) else json.dumps(value, default=str)
            flat[name] = text[:_PARAM_LIMIT]
    return flat


def log_manifests(
    tracking_uri: str,
    cfg: Settings,
    stages: tuple[str, ...] = STAGES,
    experiment: str = DEFAULT_EXPERIMENT,
) -> dict[str, str]:
    """Log one run per manifest. Returns {manifest_id: "created" | "exists"}."""
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)
    experiment_id = mlflow.set_experiment(experiment).experiment_id

    outcome: dict[str, str] = {}
    for stage in stages:
        manifest = build_manifest(stage, cfg)
        existing = client.search_runs(
            [experiment_id], filter_string=f"tags.manifest_id = '{manifest.manifest_id}'", max_results=1
        )
        if existing:
            outcome[manifest.manifest_id] = "exists"
            continue
        with mlflow.start_run(run_name=f"{stage}:{manifest.manifest_id}"):
            mlflow.set_tags({
                "manifest_id": manifest.manifest_id,
                "stage": stage,
                "model": manifest.model,
            })
            mlflow.log_params(flatten(manifest.identity))
            mlflow.log_dict(json.loads(manifest.to_json()), f"{stage}-manifest.json")
        outcome[manifest.manifest_id] = "created"
    return outcome


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--stage", action="append", choices=STAGES, help="only this stage (repeatable)")
    args = parser.parse_args(argv)

    if not args.tracking_uri:
        parser.error("give --tracking-uri or set MLFLOW_TRACKING_URI")

    outcome = log_manifests(
        args.tracking_uri, Settings(_env_file=None), tuple(args.stage or STAGES), args.experiment
    )
    for manifest_id, state in outcome.items():
        print(f"{manifest_id}  {state}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
