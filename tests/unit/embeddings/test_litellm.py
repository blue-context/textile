"""Tests for LiteLLM embedding implementation."""


import numpy as np
import pytest

from textile.embeddings.litellm import Embedding


class TestLiteLLMEmbedding:
    """Test LiteLLM embedding model."""

    def test_init_with_explicit_dimensions(self, mock_litellm):
        model = Embedding(model="text-embedding-3-small", dimensions=512)
        assert model.model == "text-embedding-3-small"
        assert model.dimension == 512

    def test_init_auto_detect_dimensions(self, mock_litellm, sample_embedding_vector):
        model = Embedding(model="text-embedding-3-small")
        assert model.dimension == len(sample_embedding_vector)

    @pytest.mark.parametrize("model_name", [
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    ])
    def test_model_name_stored(self, mock_litellm, model_name):
        model = Embedding(model=model_name, dimensions=1536)
        assert model.model == model_name

    def test_encode_returns_float32_array(self, mock_litellm, sample_text):
        model = Embedding(dimensions=1536)
        result = model.encode(sample_text)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (1536,)

    def test_encode_batch_returns_2d_array(self, mock_litellm, sample_texts):
        model = Embedding(dimensions=1536)
        result = model.encode_batch(sample_texts)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (len(sample_texts), 1536)

    @pytest.mark.parametrize("batch_size", [1, 3, 5, 10])
    def test_encode_batch_handles_different_sizes(self, mock_litellm, batch_size):
        model = Embedding(dimensions=1536)
        texts = ["text"] * batch_size
        result = model.encode_batch(texts)
        assert result.shape == (batch_size, 1536)

    def test_litellm_kwargs_passed_through(self, monkeypatch, mock_embedding_response):
        """Verify extra kwargs are passed to litellm.embedding."""
        call_kwargs = {}
        def capture_kwargs(model, input, **kwargs):
            call_kwargs.update(kwargs)
            return mock_embedding_response([0.1] * 1536)

        import litellm
        monkeypatch.setattr(litellm, "embedding", capture_kwargs)
        model = Embedding(dimensions=1536, timeout=30, max_retries=3)
        model.encode("test")
        assert call_kwargs["timeout"] == 30 and call_kwargs["max_retries"] == 3

    def test_dimension_property_consistent(self, monkeypatch):
        def mock_embedding_512(model, input, **kwargs):
            class MockResponse:
                data = [{"embedding": [0.1] * 512}]
            return MockResponse()
        import litellm
        monkeypatch.setattr(litellm, "embedding", mock_embedding_512)
        model = Embedding(dimensions=512)
        assert model.dimension == 512 and len(model.encode("test")) == 512

    def test_encode_empty_string(self, mock_litellm):
        model = Embedding(dimensions=1536)
        assert model.encode("").shape == (1536,)

    def test_encode_batch_preserves_order(self, monkeypatch, mock_embedding_response):
        """Verify batch encoding preserves input order."""
        def ordered_embedding(model, input, **kwargs):
            if isinstance(input, list):
                embeddings = [[i / 10.0] * 1536 for i in range(len(input))]
                class Response:
                    data = [{"embedding": emb} for emb in embeddings]
                return Response()
            return mock_embedding_response([0.1] * 1536)

        import litellm
        monkeypatch.setattr(litellm, "embedding", ordered_embedding)
        model = Embedding(dimensions=1536)
        result = model.encode_batch(["a", "b", "c"])
        assert result[0][0] == pytest.approx(0.0)
        assert result[1][0] == pytest.approx(0.1)
        assert result[2][0] == pytest.approx(0.2)
