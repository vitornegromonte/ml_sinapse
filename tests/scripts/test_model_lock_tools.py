import json
from pathlib import Path

import pytest

from config import Settings
from scripts import check_model_lock, update_model_lock

REPO = Path(__file__).resolve().parents[2]
SHA_A, SHA_B = "a" * 40, "b" * 40


def _settings():
    return Settings(_env_file=None)


def _write_lock(tmp_path, huggingface=None, artifacts=None, name="models.lock.json"):
    defaults = _settings()
    pins = huggingface if huggingface is not None else {
        defaults.ocr_model_name: SHA_A,
        defaults.llm_model_path: SHA_A,
        defaults.embedding_model_name: SHA_A,
    }
    path = tmp_path / name
    path.write_text(json.dumps({
        "lock_version": 1,
        "huggingface": {m: {"revision": r, "locked_at": "2026-01-01"} for m, r in pins.items()},
        "artifacts": artifacts or {},
    }))
    return path


def _digest(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


class TestCheckLock:
    def test_the_committed_lock_is_valid(self):
        """Arrange: the real artifacts/models.lock.json and the real artifacts.
        Act: check it with the default settings.
        Assert: no problems. A bad edit to the lock fails here, before CI.
        """
        from workers.linkage_scoring_worker import FEATURE_NAMES

        problems = check_model_lock.check_lock(
            REPO / "artifacts/models.lock.json", REPO / "artifacts", _settings(), len(FEATURE_NAMES)
        )

        assert problems == []

    def test_a_consistent_lock_passes(self, tmp_path):
        """Arrange: a lock that pins the three default models and hashes one artifact.
        Act: check it.
        Assert: no problems.
        """
        (tmp_path / "clf.joblib").write_bytes(b"model bytes")
        lock = _write_lock(tmp_path, artifacts={
            "clf.joblib": {"sha256": _digest(b"model bytes"), "encoder": _settings().embedding_model_name, "n_features": 7},
        })

        assert check_model_lock.check_lock(lock, tmp_path, _settings(), 7) == []

    def test_a_default_model_without_a_pin_is_a_problem(self, tmp_path):
        """Arrange: a lock that pins nothing.
        Act: check it.
        Assert: one problem per default model, naming the setting.
        """
        lock = _write_lock(tmp_path, huggingface={})

        problems = check_model_lock.check_lock(lock, tmp_path, _settings())

        assert len(problems) == 3
        assert any("ocr_model_name" in p for p in problems)
        assert any("llm_model_path" in p for p in problems)
        assert any("embedding_model_name" in p for p in problems)

    @pytest.mark.parametrize("revision", ["main", "abc123", "A" * 40, "g" * 40])
    def test_a_revision_that_is_not_a_commit_hash_is_a_problem(self, tmp_path, revision):
        """Arrange: pins where one is a branch name, a short hash, or not hex.
        Act: check the lock.
        Assert: a problem says it is not a 40-character commit hash.
        """
        defaults = _settings()
        lock = _write_lock(tmp_path, huggingface={
            defaults.ocr_model_name: revision,
            defaults.llm_model_path: SHA_A,
            defaults.embedding_model_name: SHA_A,
        })

        problems = check_model_lock.check_lock(lock, tmp_path, defaults)

        assert any("40-character commit hash" in p for p in problems)

    def test_a_changed_artifact_is_a_problem(self, tmp_path):
        """Arrange: a lock hash that no longer matches the file.
        Act: check it.
        Assert: a problem says the artifact changed and names the update script.
        """
        (tmp_path / "clf.joblib").write_bytes(b"new bytes")
        lock = _write_lock(tmp_path, artifacts={"clf.joblib": {"sha256": _digest(b"old bytes")}})

        problems = check_model_lock.check_lock(lock, tmp_path, _settings())

        assert any("clf.joblib changed" in p and "update_model_lock" in p for p in problems)

    def test_a_locked_artifact_that_is_missing_is_a_problem(self, tmp_path):
        """Arrange: a lock entry for a file that is not there.
        Act: check it.
        Assert: a problem says the file is missing.
        """
        lock = _write_lock(tmp_path, artifacts={"gone.joblib": {"sha256": "0" * 64}})

        problems = check_model_lock.check_lock(lock, tmp_path, _settings())

        assert any("gone.joblib" in p and "missing" in p for p in problems)

    def test_an_artifact_nobody_locked_is_a_problem(self, tmp_path):
        """Arrange: a .joblib file in artifacts/ that the lock does not list.
        Act: check it.
        Assert: a problem says it is not in the lock.
        """
        (tmp_path / "sneaky.joblib").write_bytes(b"x")
        lock = _write_lock(tmp_path)

        problems = check_model_lock.check_lock(lock, tmp_path, _settings())

        assert any("sneaky.joblib is not in the lock" in p for p in problems)

    def test_an_encoder_that_does_not_match_the_classifier_is_a_problem(self, tmp_path):
        """Arrange: a classifier locked for a different encoder than the default.
        Act: check it.
        Assert: a problem names both.
        """
        (tmp_path / "clf.joblib").write_bytes(b"x")
        lock = _write_lock(tmp_path, artifacts={
            "clf.joblib": {"sha256": _digest(b"x"), "encoder": "org/old-encoder"},
        })

        problems = check_model_lock.check_lock(lock, tmp_path, _settings())

        assert any("org/old-encoder" in p for p in problems)

    def test_a_feature_count_mismatch_is_a_problem(self, tmp_path):
        """Arrange: a classifier locked for 7 features and a worker that builds 6.
        Act: check it.
        Assert: a problem names both numbers.
        """
        (tmp_path / "clf.joblib").write_bytes(b"x")
        lock = _write_lock(tmp_path, artifacts={"clf.joblib": {"sha256": _digest(b"x"), "n_features": 7}})

        problems = check_model_lock.check_lock(lock, tmp_path, _settings(), feature_count=6)

        assert any("expects 7 features" in p and "builds 6" in p for p in problems)

    def test_online_mode_flags_a_commit_that_does_not_exist(self, tmp_path):
        """Arrange: a valid lock and a Hub lookup that says no commit exists.
        Act: check it online.
        Assert: one problem per pinned model.
        """
        lock = _write_lock(tmp_path)

        problems = check_model_lock.check_lock(
            lock, tmp_path, _settings(), online=True, exists=lambda model, revision: False
        )

        assert len([p for p in problems if "does not exist on Hugging Face" in p]) == 3

    def test_online_mode_accepts_commits_that_exist(self, tmp_path):
        """Arrange: a valid lock and a Hub lookup that finds every commit.
        Act: check it online.
        Assert: no problems.
        """
        lock = _write_lock(tmp_path)

        assert check_model_lock.check_lock(
            lock, tmp_path, _settings(), online=True, exists=lambda model, revision: True
        ) == []

    def test_offline_mode_never_asks_the_hub(self, tmp_path):
        """Arrange: a lookup function that fails if it is called.
        Act: check offline.
        Assert: it is never called.
        """
        def boom(model, revision):
            raise AssertionError("must not be called offline")

        check_model_lock.check_lock(_write_lock(tmp_path), tmp_path, _settings(), online=False, exists=boom)

    def test_an_invalid_lock_is_one_clear_problem(self, tmp_path):
        """Arrange: a lock file that is not JSON.
        Act: check it.
        Assert: a single problem that says the lock is invalid.
        """
        path = tmp_path / "bad.json"
        path.write_text("{nope")

        problems = check_model_lock.check_lock(path, tmp_path, _settings())

        assert len(problems) == 1 and "Invalid model lock" in problems[0]

    def test_main_exits_zero_on_the_real_lock(self, capsys):
        """Arrange: the repository as it is.
        Act: run the script's main.
        Assert: exit code 0 and an OK line.
        """
        assert check_model_lock.main([]) == 0
        assert "OK" in capsys.readouterr().out


class TestUpdateLock:
    LOCK = {
        "lock_version": 1,
        "huggingface": {"org/a": {"revision": SHA_A, "locked_at": "2026-01-01"},
                        "org/b": {"revision": SHA_A, "locked_at": "2026-01-01"}},
        "artifacts": {"clf.joblib": {"sha256": "0" * 64}},
    }

    def test_an_unchanged_lock_reports_no_changes(self, tmp_path):
        """Arrange: a lock whose models are already at the newest commit.
        Act: update it.
        Assert: no changes.
        """
        (tmp_path / "clf.joblib").write_bytes(b"x")
        lock = json.loads(json.dumps(self.LOCK))
        lock["artifacts"]["clf.joblib"]["sha256"] = _digest(b"x")

        updated, changes = update_model_lock.update_lock(lock, tmp_path, fetch=lambda model: SHA_A)

        assert changes == [] and updated == lock

    def test_a_new_commit_is_reported_and_dated(self, tmp_path):
        """Arrange: one model has a newer commit on the Hub.
        Act: update the lock for a fixed date.
        Assert: the change names both short commits, the new commit and date are
        written, and the other model is untouched.
        """
        newest = {"org/a": SHA_B, "org/b": SHA_A}

        updated, changes = update_model_lock.update_lock(
            self.LOCK, tmp_path, fetch=newest.__getitem__, today="2026-12-01"
        )

        assert any("org/a" in c and "aaaaaaaaaaaa -> bbbbbbbbbbbb" in c for c in changes)
        assert updated["huggingface"]["org/a"] == {"revision": SHA_B, "locked_at": "2026-12-01"}
        assert updated["huggingface"]["org/b"]["locked_at"] == "2026-01-01"

    def test_the_input_lock_is_not_mutated(self, tmp_path):
        """Arrange: a lock and a Hub with a newer commit.
        Act: update it.
        Assert: the original dict is unchanged.
        """
        before = json.dumps(self.LOCK, sort_keys=True)

        update_model_lock.update_lock(self.LOCK, tmp_path, fetch=lambda model: SHA_B)

        assert json.dumps(self.LOCK, sort_keys=True) == before

    def test_only_limits_the_update_to_the_named_models(self, tmp_path):
        """Arrange: two models with newer commits.
        Act: update with only=org/a.
        Assert: org/a changes, org/b and the artifacts do not.
        """
        (tmp_path / "clf.joblib").write_bytes(b"changed")

        updated, changes = update_model_lock.update_lock(
            self.LOCK, tmp_path, only={"org/a"}, fetch=lambda model: SHA_B
        )

        assert len(changes) == 1 and "org/a" in changes[0]
        assert updated["huggingface"]["org/b"]["revision"] == SHA_A
        assert updated["artifacts"]["clf.joblib"]["sha256"] == "0" * 64

    def test_a_changed_artifact_gets_a_new_hash(self, tmp_path):
        """Arrange: an artifact file whose content differs from the locked hash.
        Act: update the lock.
        Assert: the new SHA-256 is recorded.
        """
        (tmp_path / "clf.joblib").write_bytes(b"new content")

        updated, changes = update_model_lock.update_lock(self.LOCK, tmp_path, fetch=lambda model: SHA_A)

        assert updated["artifacts"]["clf.joblib"]["sha256"] == _digest(b"new content")
        assert any("clf.joblib" in c for c in changes)

    def test_a_missing_artifact_file_keeps_its_hash(self, tmp_path):
        """Arrange: a locked artifact whose file is gone.
        Act: update the lock.
        Assert: its hash is kept, so the checker still reports the missing file.
        """
        updated, _ = update_model_lock.update_lock(self.LOCK, tmp_path, fetch=lambda model: SHA_A)

        assert updated["artifacts"]["clf.joblib"]["sha256"] == "0" * 64

    def test_main_is_a_dry_run_unless_write_is_given(self, tmp_path, monkeypatch, capsys):
        """Arrange: a repo folder with a lock and a Hub with newer commits.
        Act: run main without, then with, --write.
        Assert: the file is unchanged after the dry run and updated after --write.
        """
        (tmp_path / "artifacts").mkdir()
        lock_path = tmp_path / "artifacts/models.lock.json"
        lock_path.write_text(json.dumps({**self.LOCK, "artifacts": {}}))
        monkeypatch.setattr(update_model_lock, "REPO", tmp_path)
        monkeypatch.setattr(update_model_lock, "latest_revision", lambda model: SHA_B)
        original = lock_path.read_text()

        update_model_lock.main([])
        assert lock_path.read_text() == original
        assert "Dry run" in capsys.readouterr().out

        update_model_lock.main(["--write"])
        assert json.loads(lock_path.read_text())["huggingface"]["org/a"]["revision"] == SHA_B
