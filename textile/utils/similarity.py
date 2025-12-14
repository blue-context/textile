"""Cosine similarity utility for semantic comparisons."""

import numpy as np
import numpy.typing as npt


def cosine_similarity(
    a: npt.NDArray[np.float32] | list[float],
    b: npt.NDArray[np.float32] | list[float],
) -> float:
    """Compute cosine similarity between vectors.

    Formula: cos(θ) = (a · b) / (||a|| × ||b||)

    Args:
        a: First vector (numpy array or list)
        b: Second vector (numpy array or list)

    Returns:
        Similarity score in [0, 1]:
            - 1.0: Identical vectors
            - 0.5: 60° angle
            - 0.0: Orthogonal or zero vector

    Raises:
        ValueError: If shapes differ or not 1D

    Example:
        >>> import numpy as np
        >>> a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        >>> b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        >>> cosine_similarity(a, b)
        1.0

        >>> c = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        >>> cosine_similarity(a, c)
        0.0
    """
    # Convert to numpy arrays if needed
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)

    if a_arr.shape != b_arr.shape:
        raise ValueError(f"Vectors must have same shape, got {a_arr.shape} and {b_arr.shape}")

    if a_arr.ndim != 1:
        raise ValueError(f"Vectors must be 1D, got shape {a_arr.shape}")

    dot_product = np.dot(a_arr, b_arr)

    if (norm_a := np.linalg.norm(a_arr)) == 0 or (norm_b := np.linalg.norm(b_arr)) == 0:
        return 0.0

    similarity = dot_product / (norm_a * norm_b)

    # Clamp to [0, 1] for embedding vectors (handles floating-point precision)
    # Theoretical range is [-1, 1], but embeddings typically yield [0, 1]
    return float(np.clip(similarity, 0.0, 1.0))
