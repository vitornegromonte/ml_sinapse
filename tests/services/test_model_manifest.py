import json
import shutil
from pathlib import Path

import pytest

from config import Settings
from services.model_manifest import (
    STAGES,
    build_manifest,
    load_lock,
    revision_for,
    sha256_file,
    sha256_text,
)

REPO = Path(__file__).resolve().parents[2]
CLASSIFIER = REPO / "artifacts/bo_linkage_classifier_v1.joblib"
WEIGHTS = REPO / "artifacts/bo_scoring_weights_v1.json"


def _lock(tmp_path, huggingface=None, name="lock.json", version=1):
    path = tmp_path / name
    path.write_text(json.dumps({
        "lock_version": version,
        "huggingface": {m: {"revision": r} for m, r in (huggingface or {}).items()},
        "artifacts": {},
    }))
    return str(path)


def _cfg(tmp_path, lock=None, **overrides):
    values = {
        "model_lock_path": lock or _lock(tmp_path, {
            "Dharma-AI/Dharma-OCR-LITE": "a" * 40,
            "maritaca-ai/S5-nano": "b" * 40,
            "neuralmind/bert-base-portuguese-cased": "c" * 40,
        }),
        "linkage_classifier_path": str(CLASSIFIER),
        "scoring_weights_path": str(WEIGHTS),
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


class TestLock:
    def test_pinned_revisions_are_read(self, tmp_path):
        """Arrange: a lock file that pins one model.
        Act: load it and ask for that model and an unknown one.
        Assert: the pinned commit comes back, and the unknown model gives None.
        """
        path = _lock(tmp_path, {"org/model": "f" * 40})

        lock = load_lock(path)

        assert lock.revision("org/model") == "f" * 40
        assert lock.revision("org/other") is None

    def test_missing_file_pins_nothing(self, tmp_path, caplog):
        """Arrange: a lock path that does not exist.
        Act: load it.
        Assert: an empty lock and a warning, not a crash.
        """
        with caplog.at_level("WARNING"):
            lock = load_lock(str(tmp_path / "absent.json"))

        assert lock.huggingface == {}
        assert "no model is pinned" in caplog.text

    def test_invalid_json_raises(self, tmp_path):
        """Arrange: a lock file that is not JSON.
        Act: load it.
        Assert: ValueError, so a broken lock stops the worker and does not
        silently unpin every model.
        """
        path = tmp_path / "bad.json"
        path.write_text("{not json")

        with pytest.raises(ValueError, match="Invalid model lock"):
            load_lock(str(path))

    def test_wrong_lock_version_raises(self, tmp_path):
        """Arrange: a lock file with an unknown lock_version.
        Act: load it.
        Assert: ValueError naming the version.
        """
        with pytest.raises(ValueError, match="lock_version"):
            load_lock(_lock(tmp_path, version=99, name="v99.json"))

    def test_entry_without_a_revision_raises(self, tmp_path):
        """Arrange: a pinned model entry that has no "revision" key.
        Act: load it.
        Assert: ValueError.
        """
        path = tmp_path / "norev.json"
        path.write_text(json.dumps({"lock_version": 1, "huggingface": {"m": {"locked_at": "x"}}}))

        with pytest.raises(ValueError, match="Invalid model lock"):
            load_lock(str(path))

    def test_revision_for_reads_the_configured_lock(self, tmp_path):
        """Arrange: settings that point at a lock pinning one model.
        Act: ask revision_for that model.
        Assert: the pinned commit.
        """
        cfg = _cfg(tmp_path, lock=_lock(tmp_path, {"org/m": "9" * 40}, name="rf.json"))

        assert revision_for("org/m", cfg) == "9" * 40


class TestManifestIdentity:
    @pytest.mark.parametrize("stage", STAGES)
    def test_same_settings_give_the_same_id(self, tmp_path, stage):
        """Arrange: one set of settings.
        Act: build the manifest of a stage twice.
        Assert: the same manifest_id.
        """
        cfg = _cfg(tmp_path)

        assert build_manifest(stage, cfg).manifest_id == build_manifest(stage, cfg).manifest_id

    @pytest.mark.parametrize("stage, change", [
        ("ocr", {"ocr_render_dpi": 300}),
        ("ocr", {"ocr_max_tokens": 2048}),
        ("llm", {"llm_sampling_params": {"max_tokens": 64}}),
        ("llm", {"llm_chat_template_kwargs": {"enable_thinking": True}}),
        ("llm", {"llm_model_path": "org/other-model"}),
        ("embedding", {"embedding_model_name": "org/other-encoder"}),
        ("linkage", {"linkage_tau_dias": 7.0}),
        ("linkage", {"linkage_limiar_descricao": 0.5}),
        ("cluster", {"cluster_score_threshold": 0.8}),
    ])
    def test_a_setting_that_changes_the_output_changes_the_id(self, tmp_path, stage, change):
        """Arrange: a base manifest, and settings with one output-changing value edited.
        Act: build the manifest of the same stage again.
        Assert: the manifest_id differs.
        """
        base = build_manifest(stage, _cfg(tmp_path))

        changed = build_manifest(stage, _cfg(tmp_path, **change))

        assert changed.manifest_id != base.manifest_id

    def test_a_setting_that_does_not_change_the_output_keeps_the_id(self, tmp_path):
        """Arrange: a base LLM manifest.
        Act: change only the GPU share and the engine arguments.
        Assert: the id is the same, because memory tuning does not change the text.
        """
        base = build_manifest("llm", _cfg(tmp_path))

        tuned = build_manifest("llm", _cfg(
            tmp_path, llm_gpu_memory_utilization=0.4, llm_engine_kwargs={"max_num_seqs": 8}
        ))

        assert tuned.manifest_id == base.manifest_id

    def test_a_new_prompt_changes_the_llm_id(self, tmp_path):
        """Arrange: a base LLM manifest.
        Act: change the prompt template.
        Assert: a different id and a different prompt_sha.
        """
        base = build_manifest("llm", _cfg(tmp_path))

        changed = build_manifest("llm", _cfg(tmp_path, llm_prompt_template="Resuma: {text}"))

        assert changed.manifest_id != base.manifest_id
        assert changed.identity["prompt_sha"] != base.identity["prompt_sha"]

    def test_a_new_pin_changes_the_id(self, tmp_path):
        """Arrange: two locks that pin the same model to two commits.
        Act: build the embedding manifest under each.
        Assert: the ids differ, so an upgrade of the weights is visible.
        """
        old = _cfg(tmp_path, lock=_lock(tmp_path, {"neuralmind/bert-base-portuguese-cased": "1" * 40}, name="old.json"))
        new = _cfg(tmp_path, lock=_lock(tmp_path, {"neuralmind/bert-base-portuguese-cased": "2" * 40}, name="new.json"))

        assert build_manifest("embedding", old).manifest_id != build_manifest("embedding", new).manifest_id

    def test_pinned_and_unpinned_models_are_labelled(self, tmp_path):
        """Arrange: a lock that pins the OCR model and none for the LLM.
        Act: build both manifests.
        Assert: the OCR label has the 12-character commit, the LLM label says unpinned.
        """
        cfg = _cfg(tmp_path, lock=_lock(tmp_path, {"Dharma-AI/Dharma-OCR-LITE": "d" * 40}, name="lbl.json"))

        assert build_manifest("ocr", cfg).model == "Dharma-AI/Dharma-OCR-LITE@" + "d" * 12
        assert build_manifest("llm", cfg).model == "maritaca-ai/S5-nano (unpinned)"

    def test_linkage_id_follows_the_encoder(self, tmp_path):
        """Arrange: a base linkage manifest.
        Act: change only the embedding model name.
        Assert: the linkage id changes, because a score is valid only for the
        encoder it was calibrated with.
        """
        base = build_manifest("linkage", _cfg(tmp_path))

        changed = build_manifest("linkage", _cfg(tmp_path, embedding_model_name="org/other-encoder"))

        assert changed.manifest_id != base.manifest_id
        assert changed.identity["encoder_manifest_id"] != base.identity["encoder_manifest_id"]

    def test_linkage_id_follows_the_classifier_file(self, tmp_path):
        """Arrange: a copy of the classifier file, and a copy with one byte changed.
        Act: build the linkage manifest for each.
        Assert: the ids differ, and the label carries the file hash.
        """
        original = tmp_path / "clf.joblib"
        shutil.copy(CLASSIFIER, original)
        edited = tmp_path / "clf2.joblib"
        edited.write_bytes(CLASSIFIER.read_bytes() + b"\x00")

        one = build_manifest("linkage", _cfg(tmp_path, linkage_classifier_path=str(original)))
        two = build_manifest("linkage", _cfg(tmp_path, linkage_classifier_path=str(edited)))

        assert one.manifest_id != two.manifest_id
        assert one.model == f"bo_linkage_classifier@{sha256_file(original)[:12]}"

    def test_a_missing_classifier_is_visible_in_the_label(self, tmp_path):
        """Arrange: settings that point at a classifier file that does not exist.
        Act: build the linkage manifest.
        Assert: the label says "missing", so a bad deploy is not hidden.
        """
        manifest = build_manifest("linkage", _cfg(tmp_path, linkage_classifier_path=str(tmp_path / "none.joblib")))

        assert manifest.model == "bo_linkage_classifier@missing"

    def test_scoring_id_follows_the_weights_file(self, tmp_path):
        """Arrange: a copy of the weights with a different version.
        Act: build the scoring manifest for each.
        Assert: the ids differ and the label shows the version.
        """
        data = json.loads(WEIGHTS.read_text())
        edited = tmp_path / "weights.json"
        edited.write_text(json.dumps({**data, "version": "v2"}))

        base = build_manifest("scoring", _cfg(tmp_path))
        changed = build_manifest("scoring", _cfg(tmp_path, scoring_weights_path=str(edited)))

        assert changed.manifest_id != base.manifest_id
        assert changed.model == "bo_maturity_score@v2"

    def test_ocr_manifest_records_the_prompts_and_the_repetition_penalty(self, tmp_path):
        """Arrange: settings for Dharma and for GLM.
        Act: build the OCR manifest of each.
        Assert: the prompts differ, and only Dharma records its repetition penalty.
        """
        dharma = build_manifest("ocr", _cfg(tmp_path))
        glm = build_manifest("ocr", _cfg(tmp_path, ocr_model_name="zai-org/GLM-OCR"))

        assert dharma.identity["repetition_penalty"] == 1.05
        assert glm.identity["repetition_penalty"] == 1.0
        assert dharma.identity["prompts_sha"] != glm.identity["prompts_sha"]

    def test_ocr_from_mlflow_uses_the_uri_and_no_pin(self, tmp_path):
        """Arrange: settings with an MLflow model URI.
        Act: build the OCR manifest.
        Assert: the model is the URI and no Hugging Face revision is recorded.
        """
        manifest = build_manifest("ocr", _cfg(tmp_path, ocr_model_uri="models:/ocr/3"))

        assert manifest.identity["model"] == "models:/ocr/3"
        assert manifest.identity["revision"] is None

    def test_unknown_stage_raises(self, tmp_path):
        """Arrange: a stage name that does not exist.
        Act: build its manifest.
        Assert: ValueError listing the valid stages.
        """
        with pytest.raises(ValueError, match="Unknown stage"):
            build_manifest("tokenizer", _cfg(tmp_path))

    def test_lineage_and_json_carry_the_id(self, tmp_path):
        """Arrange: an LLM manifest.
        Act: read its lineage fields and its JSON.
        Assert: the lineage has exactly manifest_id and model, and the JSON
        parses and repeats the id.
        """
        manifest = build_manifest("llm", _cfg(tmp_path))

        assert set(manifest.lineage()) == {"manifest_id", "model"}
        assert json.loads(manifest.to_json())["manifest_id"] == manifest.manifest_id


class TestHashes:
    def test_sha256_file_matches_hashlib_and_flags_a_missing_file(self, tmp_path):
        """Arrange: a file with known content, and a path that does not exist.
        Act: hash both.
        Assert: the known digest, and "missing" for the absent file.
        """
        file = tmp_path / "f.bin"
        file.write_bytes(b"abc")

        assert sha256_file(file) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        assert sha256_file(tmp_path / "nope") == "missing"

    def test_sha256_text_separates_its_parts(self):
        """Arrange: two ways to split the same letters into parts.
        Act: hash both.
        Assert: the hashes differ, so ("ab", "c") is not confused with ("a", "bc").
        """
        assert sha256_text("ab", "c") != sha256_text("a", "bc")


class TestPublishManifest:
    def test_the_manifest_document_is_saved_under_its_id(self, tmp_path):
        """Arrange: an LLM manifest and a storage mock.
        Act: publish it.
        Assert: it is saved with the prefix "manifests", its id as the key, and
        JSON that parses back to the same id.
        """
        from unittest.mock import MagicMock

        from services.model_manifest import publish_manifest

        manifest = build_manifest("llm", _cfg(tmp_path))
        storage = MagicMock()

        publish_manifest(storage, manifest)

        kwargs = storage.save.call_args.kwargs
        assert kwargs["prefix"] == "manifests"
        assert kwargs["document_id"] == manifest.manifest_id
        assert kwargs["content_type"] == "application/json"
        assert json.loads(kwargs["data"])["manifest_id"] == manifest.manifest_id

    def test_a_storage_fault_is_logged_and_not_raised(self, tmp_path, caplog):
        """Arrange: a storage mock that fails.
        Act: publish a manifest.
        Assert: no exception, and a warning, since the outputs carry the id anyway.
        """
        from unittest.mock import MagicMock

        from services.model_manifest import publish_manifest

        storage = MagicMock()
        storage.save.side_effect = ConnectionError("minio down")

        with caplog.at_level("WARNING"):
            publish_manifest(storage, build_manifest("llm", _cfg(tmp_path)))

        assert "Could not publish manifest" in caplog.text
