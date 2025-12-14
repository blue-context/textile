"""Shared fixtures for transformer tests."""

import numpy as np
import pytest

from textile.core.context_window import ContextWindow
from textile.core.message import Message
from textile.core.turn_state import TurnState


def create_message(role: str, content: str, turn_index: int = 0, embedding=None):
    """Helper to create message with turn_index and optional embedding."""
    msg = Message(role=role, content=content)
    msg.turn_index = turn_index
    if embedding is not None:
        # Convert numpy array to list for storage (like what would happen in practice)
        if hasattr(embedding, 'tolist'):
            msg.embedding = embedding.tolist()
        else:
            msg.embedding = embedding
    return msg


@pytest.fixture
def sample_messages():
    """Create sample messages with varied roles and turn indices."""
    return [
        Message(role="system", content="You are a helpful assistant"),
        Message(role="user", content="First question"),
        Message(role="assistant", content="First answer"),
        Message(role="user", content="Second question"),
        Message(role="assistant", content="Second answer"),
    ]


@pytest.fixture
def sample_context(sample_messages):
    """Create sample ContextWindow."""
    for i, msg in enumerate(sample_messages):
        msg.turn_index = i
    return ContextWindow(messages=sample_messages.copy(), max_tokens=4096)


@pytest.fixture
def sample_state():
    """Create sample TurnState."""
    return TurnState(user_message="Hello", turn_index=5)


@pytest.fixture
def sample_embedding():
    """Create sample embedding vector."""
    return np.random.rand(384).tolist()


@pytest.fixture
def messages_with_embeddings(sample_messages, sample_embedding):
    """Create messages with embeddings."""
    msgs = sample_messages.copy()
    for msg in msgs:
        msg.embedding = np.random.rand(384).tolist()
    return msgs


@pytest.fixture
def context_with_embeddings(messages_with_embeddings):
    """Create context with embedded messages."""
    for i, msg in enumerate(messages_with_embeddings):
        msg.turn_index = i
    return ContextWindow(messages=messages_with_embeddings, max_tokens=4096)


@pytest.fixture
def state_with_embedding(sample_embedding):
    """Create state with user embedding."""
    return TurnState(
        user_message="Test query",
        turn_index=5,
        user_embedding=sample_embedding,
    )


@pytest.fixture
def sample_tools():
    """Create sample tool definitions."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather forecast for a location",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_database",
                "description": "Search the knowledge database",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "send_email",
                "description": "Send an email message",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]
