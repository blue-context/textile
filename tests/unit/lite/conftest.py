"""Shared fixtures for lite module tests."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


@pytest.fixture
def mock_completion_response():
    """Mock LiteLLM completion response."""
    message = SimpleNamespace(content="Test response")
    choice = SimpleNamespace(message=message, index=0, finish_reason="stop")
    return SimpleNamespace(
        choices=[choice],
        model="gpt-4",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )


@pytest.fixture
def mock_streaming_chunk():
    """Mock LiteLLM streaming chunk."""
    delta = SimpleNamespace(content="chunk")
    choice = SimpleNamespace(delta=delta, index=0, finish_reason=None)
    return SimpleNamespace(choices=[choice])


@pytest.fixture
def mock_embedding_response():
    """Mock LiteLLM embedding response."""
    embedding_item = SimpleNamespace(embedding=[0.1, 0.2, 0.3], index=0)
    return SimpleNamespace(
        data=[embedding_item],
        object="list",
        usage=SimpleNamespace(prompt_tokens=5, total_tokens=5),
    )


@pytest.fixture
def sample_messages():
    """Sample message list for testing."""
    return [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
    ]


@pytest.fixture
def mock_transformer():
    """Mock transformer with transform and on_response methods."""
    transformer = Mock()
    transformer.transform.return_value = (Mock(), Mock())
    transformer.on_response.return_value = []
    return transformer


@pytest.fixture
def mock_async_store():
    """Mock async storage."""
    store = AsyncMock()
    store.store_embedding_event = AsyncMock(return_value=None)
    return store


@pytest.fixture
def mock_sync_store():
    """Mock sync storage."""
    store = Mock()
    store.store_embedding_event = Mock(return_value=None)
    return store


@pytest.fixture
def capture_transformer():
    """Transformer that captures context and state."""
    captured = {"context": None, "state": None}

    class CaptureTransformer:
        def transform(self, context, state):
            captured["context"] = context
            captured["state"] = state
            return context, state

    return CaptureTransformer(), captured
