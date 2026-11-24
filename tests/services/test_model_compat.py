import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from config import Settings
from services.model_compat import (
    ModelCompatibilityError,
    check_classifier_file,
    check_encoder_matches_classifier,
    check_loaded_classifier,
)

REPO = Path(__file__).resolve().parents[2]
LOCK = str(REPO / "artifacts/models.lock.json")
CLASSIFIER = REPO / "artifacts/bo_linkage_classifier_v1.joblib"
NAMES = ("a", "b", "c", "d", "e", "f", "g")


def _cfg(**overrides):
    values = {"model_lock_path": LOCK, "linkage_classifier_path": str(CLASSIFIER)}
    values.update(overrides)
    return Settings(_env_file=None, **values)


class TestEncoderMatchesClassifier:
    def test_the_calibrated_encoder_passes(self):
        """Arrange: the default encoder, which the classifier was calibrated with.
        Act: check it.
        Assert: no error.
        """
        check_encoder_matches_classifier(_cfg())

    def test_a_different_encoder_is_refused(self):
        """Arrange: settings with another encoder.
        Act: check it.
        Assert: ModelCompatibilityError naming both encoders.
        """
        with pytest.raises(ModelCompatibilityError, match="calibrated with"):
            check_encoder_matches_classifier(_cfg(embedding_model_name="org/other-encoder"))

    def test_the_escape_hatch_turns_the_error_into_a_warning(self, caplog):
        """Arrange: another encoder, with allow_incompatible_models on.
        Act: check it.
        Assert: no error, and a warning says the mismatch was allowed.
        """
        with caplog.at_level("WARNING"):
            check_encoder_matches_classifier(
                _cfg(embedding_model_name="org/other-encoder", allow_incompatible_models=True)
            )

        assert "allowed by ALLOW_INCOMPATIBLE_MODELS" in caplog.text

    def test_a_classifier_that_is_not_locked_is_not_checked(self, tmp_path):
        """Arrange: a classifier file name that the lock does not know.
        Act: check an unrelated encoder.
        Assert: no error, since there is nothing to compare with.
        """
        check_encoder_matches_classifier(
            _cfg(linkage_classifier_path=str(tmp_path / "other.joblib"), embedding_model_name="org/x")
        )


class TestClassifierFile:
    def test_the_locked_file_passes(self):
        """Arrange: the real classifier file.
        Act: check it.
        Assert: no error.
        """
        check_classifier_file(_cfg())

    def test_a_changed_file_is_refused(self, tmp_path):
        """Arrange: a copy of the classifier with one byte added, under the locked name.
        Act: check it.
        Assert: ModelCompatibilityError that names the lock.
        """
        edited = tmp_path / CLASSIFIER.name
        edited.write_bytes(CLASSIFIER.read_bytes() + b"\x00")

        with pytest.raises(ModelCompatibilityError, match="models.lock.json"):
            check_classifier_file(_cfg(linkage_classifier_path=str(edited)))

    def test_an_identical_copy_passes(self, tmp_path):
        """Arrange: a byte-identical copy of the classifier in another folder.
        Act: check it.
        Assert: no error, since only the content matters.
        """
        copy = tmp_path / CLASSIFIER.name
        shutil.copy(CLASSIFIER, copy)

        check_classifier_file(_cfg(linkage_classifier_path=str(copy)))

    def test_a_missing_file_is_skipped(self, tmp_path):
        """Arrange: the locked name, but no file there (a developer machine).
        Act: check it.
        Assert: no error. Loading the file fails on its own later.
        """
        check_classifier_file(_cfg(linkage_classifier_path=str(tmp_path / CLASSIFIER.name)))

    def test_the_escape_hatch_allows_a_changed_file(self, tmp_path, caplog):
        """Arrange: a changed file with allow_incompatible_models on.
        Act: check it.
        Assert: no error and a warning.
        """
        edited = tmp_path / CLASSIFIER.name
        edited.write_bytes(CLASSIFIER.read_bytes() + b"\x00")

        with caplog.at_level("WARNING"):
            check_classifier_file(_cfg(linkage_classifier_path=str(edited), allow_incompatible_models=True))

        assert "allowed by ALLOW_INCOMPATIBLE_MODELS" in caplog.text


class TestLoadedClassifier:
    def test_matching_feature_count_passes(self):
        """Arrange: a classifier that expects 7 features.
        Act: check it against the worker's 7 feature names.
        Assert: no error.
        """
        check_loaded_classifier(_cfg(), MagicMock(n_features_in_=7), NAMES)

    @pytest.mark.parametrize("count", [6, 8, None])
    def test_other_feature_counts_are_refused(self, count):
        """Arrange: a classifier that expects another number of features, or none.
        Act: check it.
        Assert: ModelCompatibilityError that names the feature list.
        """
        with pytest.raises(ModelCompatibilityError, match="features"):
            check_loaded_classifier(_cfg(), MagicMock(n_features_in_=count), NAMES)
