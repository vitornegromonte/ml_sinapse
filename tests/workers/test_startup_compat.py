"""Workers that depend on the linkage encoder refuse to start with another one."""
import pytest

from config import Settings
from services.model_compat import ModelCompatibilityError
from workers.embedding_worker import EmbeddingWorker
from workers.linkage_scoring_worker import LinkageScoringWorker

OTHER = {"embedding_model_name": "org/other-encoder"}


@pytest.mark.parametrize("worker_class", [EmbeddingWorker, LinkageScoringWorker])
def test_a_worker_refuses_an_encoder_the_classifier_was_not_calibrated_with(worker_class):
    """Arrange: settings with an encoder other than the calibrated one.
    Act: build the embedding worker and the linkage worker.
    Assert: both raise ModelCompatibilityError, so no wrong vector or score is written.
    """
    with pytest.raises(ModelCompatibilityError):
        worker_class(Settings(_env_file=None, **OTHER))


@pytest.mark.parametrize("worker_class", [EmbeddingWorker, LinkageScoringWorker])
def test_a_worker_starts_with_the_calibrated_encoder(worker_class):
    """Arrange: the default settings.
    Act: build each worker.
    Assert: it starts, with a manifest.
    """
    worker = worker_class(Settings(_env_file=None))

    assert worker._manifest.manifest_id


@pytest.mark.parametrize("worker_class", [EmbeddingWorker, LinkageScoringWorker])
def test_the_escape_hatch_lets_a_worker_start(worker_class):
    """Arrange: another encoder with allow_incompatible_models on.
    Act: build each worker.
    Assert: it starts.
    """
    worker_class(Settings(_env_file=None, allow_incompatible_models=True, **OTHER))
