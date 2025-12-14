"""Shared fixtures for embeddings module tests."""

import pytest


@pytest.fixture
def mock_embedding_response():
    """Mock litellm embedding response structure.

    Returns:
        Mock object simulating litellm.embedding() response
    """
    class MockData:
        def __init__(self, embedding):
            self.data = [{"embedding": embedding}]

    return MockData


@pytest.fixture
def sample_embedding_vector() -> list[float]:
    """Sample embedding vector for testing.

    Returns:
        1536-dimensional embedding vector (OpenAI standard)
    """
    return [0.1] * 1536


@pytest.fixture
def sample_text() -> str:
    """Sample text for embedding tests.

    Returns:
        Simple test string
    """
    return "Hello world"


@pytest.fixture
def sample_texts() -> list[str]:
    """Sample text batch for embedding tests.

    Returns:
        List of test strings
    """
    return ["First text", "Second text", "Third text"]


@pytest.fixture
def mock_litellm(monkeypatch, sample_embedding_vector):
    """Mock litellm.embedding function.

    Returns:
        Mock function that returns sample embedding data
    """
    def mock_embedding(model: str, input: str | list[str], **kwargs):
        class MockResponse:
            def __init__(self, embeddings):
                self.data = [{"embedding": emb} for emb in embeddings]

        if isinstance(input, list):
            return MockResponse([sample_embedding_vector] * len(input))
        return MockResponse([sample_embedding_vector])

    import litellm
    monkeypatch.setattr(litellm, "embedding", mock_embedding)
    return mock_embedding
