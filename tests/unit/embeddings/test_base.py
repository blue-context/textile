"""Tests for EmbeddingModel protocol."""

import pytest
import numpy as np

from textile.embeddings.base import EmbeddingModel


class ConcreteEmbedding(EmbeddingModel):
    """Concrete implementation for testing protocol."""

    def __init__(self, dim: int = 128):
        self._dim = dim

    def encode(self, text: str) -> np.ndarray:
        """Return fixed-dimension vector."""
        return np.ones(self._dim, dtype=np.float32)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Return batch of fixed-dimension vectors."""
        return np.ones((len(texts), self._dim), dtype=np.float32)

    @property
    def dimension(self) -> int:
        """Return configured dimension."""
        return self._dim


class TestEmbeddingModel:
    """Test EmbeddingModel protocol implementation."""

    @pytest.mark.parametrize("dim", [128, 256, 512, 1536])
    def test_encode_returns_correct_shape(self, dim):
        model = ConcreteEmbedding(dim=dim)
        result = model.encode("test")
        assert result.shape == (dim,)
        assert result.dtype == np.float32

    @pytest.mark.parametrize("batch_size,dim", [
        (1, 128), (3, 256), (10, 512)
    ])
    def test_encode_batch_returns_correct_shape(self, batch_size, dim):
        model = ConcreteEmbedding(dim=dim)
        texts = ["test"] * batch_size
        result = model.encode_batch(texts)
        assert result.shape == (batch_size, dim)
        assert result.dtype == np.float32

    def test_dimension_property_returns_int(self):
        model = ConcreteEmbedding(dim=512)
        assert isinstance(model.dimension, int)
        assert model.dimension == 512

    @pytest.mark.parametrize("dim", [64, 128, 256])
    def test_encode_batch_consistent_with_encode(self, dim):
        model = ConcreteEmbedding(dim=dim)
        single = model.encode("test")
        batch = model.encode_batch(["test"])
        assert batch.shape == (1, dim)
        np.testing.assert_array_equal(single, batch[0])

    def test_protocol_methods_exist(self):
        """Verify all protocol methods are implemented."""
        model = ConcreteEmbedding()
        assert hasattr(model, "encode")
        assert hasattr(model, "encode_batch")
        assert hasattr(model, "dimension")
        assert callable(model.encode)
        assert callable(model.encode_batch)

    def test_encode_batch_empty_list(self):
        model = ConcreteEmbedding(dim=128)
        result = model.encode_batch([])
        assert result.shape == (0, 128)
