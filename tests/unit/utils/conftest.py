"""Shared fixtures for utils module tests."""

import asyncio

import numpy as np
import pytest


@pytest.fixture
def identical_vectors() -> tuple[np.ndarray, np.ndarray]:
    """Identical vectors for similarity testing.

    Returns:
        Tuple of two identical float32 arrays
    """
    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return vec, vec.copy()


@pytest.fixture
def orthogonal_vectors() -> tuple[np.ndarray, np.ndarray]:
    """Orthogonal vectors for similarity testing.

    Returns:
        Tuple of two perpendicular float32 arrays
    """
    vec_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vec_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return vec_a, vec_b


@pytest.fixture
def zero_vector() -> np.ndarray:
    """Zero vector for edge case testing.

    Returns:
        Zero-filled float32 array
    """
    return np.array([0.0, 0.0, 0.0], dtype=np.float32)


@pytest.fixture
def sample_coroutine():
    """Sample async coroutine for testing.

    Returns:
        Async function that returns a value
    """
    async def async_func(value: int) -> int:
        await asyncio.sleep(0.001)
        return value * 2

    return async_func


@pytest.fixture
def failing_coroutine():
    """Async coroutine that raises an error.

    Returns:
        Async function that raises ValueError
    """
    async def async_fail() -> None:
        await asyncio.sleep(0.001)
        raise ValueError("Test error")

    return async_fail
