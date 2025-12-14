"""Tests for cosine similarity utility."""

import pytest
import numpy as np

from textile.utils.similarity import cosine_similarity


class TestCosineSimilarity:
    """Test cosine similarity computation."""

    def test_identical_vectors_return_one(self, identical_vectors):
        a, b = identical_vectors
        assert cosine_similarity(a, b) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self, orthogonal_vectors):
        a, b = orthogonal_vectors
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    @pytest.mark.parametrize("vec", [
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.5, 0.5, 0.5], dtype=np.float32),
    ])
    def test_vector_with_itself_returns_one(self, vec):
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    @pytest.mark.parametrize("a,b,expected", [
        ([1.0, 0.0], [1.0, 0.0], 1.0),
        ([1.0, 0.0], [0.0, 1.0], 0.0),
        ([1.0, 1.0], [1.0, 1.0], 1.0),
        ([1.0, 1.0], [-1.0, -1.0], 0.0),  # Clipped to 0
    ])
    def test_known_similarities(self, a, b, expected):
        result = cosine_similarity(a, b)
        assert result == pytest.approx(expected, abs=1e-6)

    def test_zero_vector_returns_zero(self, zero_vector):
        other = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_similarity(zero_vector, other) == 0.0
        assert cosine_similarity(other, zero_vector) == 0.0

    def test_both_zero_vectors_return_zero(self, zero_vector):
        assert cosine_similarity(zero_vector, zero_vector) == 0.0

    def test_accepts_python_lists(self):
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(1.0)

    def test_converts_to_float32(self):
        a = [1, 2, 3]  # integers
        b = [1, 2, 3]
        result = cosine_similarity(a, b)
        assert result == pytest.approx(1.0)

    @pytest.mark.parametrize("a_shape,b_shape", [
        ([1, 2], [1, 2, 3]),
        ([1], [1, 2]),
        ([1, 2, 3], [1, 2]),
    ])
    def test_different_shapes_raise_error(self, a_shape, b_shape):
        a = np.array(a_shape, dtype=np.float32)
        b = np.array(b_shape, dtype=np.float32)
        with pytest.raises(ValueError, match="same shape"):
            cosine_similarity(a, b)

    @pytest.mark.parametrize("shape", [
        (2, 2),
        (3, 3, 3),
        (1, 5),
    ])
    def test_non_1d_vectors_raise_error(self, shape):
        a = np.ones(shape, dtype=np.float32)
        b = np.ones(shape, dtype=np.float32)
        with pytest.raises(ValueError, match="must be 1D"):
            cosine_similarity(a, b)

    def test_result_clamped_to_zero_one(self):
        """Verify result is always in [0, 1] range."""
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        result = cosine_similarity(a, b)
        assert 0.0 <= result <= 1.0

    @pytest.mark.parametrize("dim", [2, 10, 100, 1536])
    def test_various_dimensions(self, dim):
        a = np.random.rand(dim).astype(np.float32)
        b = np.random.rand(dim).astype(np.float32)
        result = cosine_similarity(a, b)
        assert 0.0 <= result <= 1.0
        assert isinstance(result, float)
