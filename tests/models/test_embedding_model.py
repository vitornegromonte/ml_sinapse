import sys
from unittest.mock import MagicMock, patch

from models.embedding_model import SentenceTransformerEncoder


class TestEncoderRevisionPin:
    def test_load_pins_the_revision(self):
        """Arrange: an encoder with a pinned revision and a stand-in
        sentence_transformers module.
        Act: load it.
        Assert: SentenceTransformer is called with that revision.
        """
        stub = MagicMock()
        with patch.dict(sys.modules, {"sentence_transformers": stub}):
            SentenceTransformerEncoder("org/encoder", device="cpu", revision="e" * 40).load()

        stub.SentenceTransformer.assert_called_once_with(
            "org/encoder", device="cpu", revision="e" * 40
        )

    def test_load_without_a_revision_adds_no_pin(self):
        """Arrange: an encoder with no revision.
        Act: load it.
        Assert: SentenceTransformer is called as before, with no revision.
        """
        stub = MagicMock()
        with patch.dict(sys.modules, {"sentence_transformers": stub}):
            SentenceTransformerEncoder("org/encoder", device="cpu").load()

        stub.SentenceTransformer.assert_called_once_with("org/encoder", device="cpu")
