import json
from pathlib import Path

import pytest

from config import Settings
from scripts import log_manifests_to_mlflow as logger_script
from scripts.log_manifests_to_mlflow import flatten, log_manifests
from services.model_manifest import STAGES, build_manifest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def cfg():
    return Settings(
        _env_file=None,
        model_lock_path=str(REPO / "artifacts/models.lock.json"),
        linkage_classifier_path=str(REPO / "artifacts/bo_linkage_classifier_v1.joblib"),
        scoring_weights_path=str(REPO / "artifacts/bo_scoring_weights_v1.json"),
    )


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A throwaway SQLite MLflow store. MLflow's own file store is in maintenance
    mode. Artifacts of a SQLite store go to ./mlartifacts, so work in tmp_path."""
    monkeypatch.chdir(tmp_path)
    return f"sqlite:///{tmp_path / 'mlflow.db'}"


def _runs(store, experiment="ml-sinapse/manifests"):
    from mlflow import MlflowClient

    client = MlflowClient(store)
    experiment_id = client.get_experiment_by_name(experiment).experiment_id
    return client, client.search_runs([experiment_id])


class TestFlatten:
    def test_nested_dicts_become_dotted_keys(self):
        """Arrange: an identity with a nested dict, a list, a number and a None.
        Act: flatten it.
        Assert: dotted keys, and every value is a string.
        """
        flat = flatten({"a": 1, "b": {"c": "x", "d": [1, 2]}, "e": None})

        assert flat == {"a": "1", "b.c": "x", "b.d": "[1, 2]", "e": "null"}

    def test_a_long_value_is_cut_to_the_parameter_limit(self):
        """Arrange: a value longer than the MLflow parameter limit.
        Act: flatten it.
        Assert: it is cut to 500 characters.
        """
        assert len(flatten({"k": "x" * 2000})["k"]) == 500


class TestLogManifests:
    def test_one_run_is_created_per_stage(self, store, cfg):
        """Arrange: an empty MLflow store.
        Act: log the manifests of every stage.
        Assert: one run per stage, each tagged with its manifest_id, stage and model.
        """
        outcome = log_manifests(store, cfg)

        _, runs = _runs(store)
        assert set(outcome.values()) == {"created"}
        assert {r.data.tags["stage"] for r in runs} == set(STAGES)
        for run in runs:
            manifest = build_manifest(run.data.tags["stage"], cfg)
            assert run.data.tags["manifest_id"] == manifest.manifest_id
            assert run.data.tags["model"] == manifest.model

    def test_parameters_are_the_flattened_identity(self, store, cfg):
        """Arrange: an empty store.
        Act: log only the LLM stage.
        Assert: its run has the sampling and the prompt hash as parameters.
        """
        log_manifests(store, cfg, stages=("llm",))

        _, runs = _runs(store)
        params = runs[0].data.params
        assert params["prompt_sha"] == build_manifest("llm", cfg).identity["prompt_sha"]
        assert params["sampling_params.max_tokens"] == "2048"
        assert params["revision"] == "fe3a3581c33cb22917770ef1987b7c6e30929425"

    def test_the_manifest_json_is_an_artifact(self, store, cfg):
        """Arrange: an empty store.
        Act: log the OCR stage.
        Assert: the run has an ocr-manifest.json artifact with the same manifest_id.
        """
        log_manifests(store, cfg, stages=("ocr",))

        client, runs = _runs(store)
        path = client.download_artifacts(runs[0].info.run_id, "ocr-manifest.json")
        assert json.loads(Path(path).read_text())["manifest_id"] == build_manifest("ocr", cfg).manifest_id

    def test_running_twice_creates_no_duplicates(self, store, cfg):
        """Arrange: a store that already holds every manifest.
        Act: log them again.
        Assert: every manifest is reported as existing and the run count is unchanged.
        """
        log_manifests(store, cfg)
        _, first = _runs(store)

        second = log_manifests(store, cfg)

        _, after = _runs(store)
        assert set(second.values()) == {"exists"}
        assert len(after) == len(first)

    def test_a_changed_setting_creates_a_new_run(self, store, cfg):
        """Arrange: a store holding the current LLM manifest.
        Act: change a sampling value and log again.
        Assert: the new manifest_id gets its own run, so both recipes are kept.
        """
        log_manifests(store, cfg, stages=("llm",))
        changed = cfg.model_copy(update={"llm_sampling_params": {"max_tokens": 64}})

        outcome = log_manifests(store, changed, stages=("llm",))

        _, runs = _runs(store)
        assert list(outcome.values()) == ["created"]
        assert len(runs) == 2


class TestMain:
    def test_main_needs_a_tracking_uri(self, monkeypatch):
        """Arrange: no --tracking-uri and no MLFLOW_TRACKING_URI.
        Act: run main.
        Assert: argparse exits with an error.
        """
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

        with pytest.raises(SystemExit):
            logger_script.main([])

    def test_main_logs_the_requested_stage_and_prints_the_ids(self, store, capsys):
        """Arrange: a store and one requested stage.
        Act: run main.
        Assert: exit 0 and the manifest id is printed as created.
        """
        assert logger_script.main(["--tracking-uri", store, "--stage", "cluster"]) == 0

        assert "created" in capsys.readouterr().out
