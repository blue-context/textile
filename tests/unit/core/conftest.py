"""Shared fixtures for core module tests."""

import pytest

from textile.core.context_window import ContextWindow
from textile.core.message import Message
from textile.core.metadata import MessageMetadata
from textile.core.turn_state import TurnState


@pytest.fixture
def sample_message_dict() -> dict:
    """Sample message dictionary for testing.

    Returns:
        Dict with basic message fields (role, content)
    """
    return {
        "role": "user",
        "content": "Hello world",
    }


@pytest.fixture
def sample_assistant_dict() -> dict:
    """Sample assistant message dictionary.

    Returns:
        Dict with assistant message fields
    """
    return {
        "role": "assistant",
        "content": "I can help with that!",
    }


@pytest.fixture
def sample_message(sample_message_dict: dict) -> Message:
    """Sample Message object for testing.

    Args:
        sample_message_dict: Message dictionary fixture

    Returns:
        Message instance
    """
    return Message.from_dict(sample_message_dict)


@pytest.fixture
def sample_messages() -> list[dict]:
    """Sample message list for testing conversations.

    Returns:
        List of message dictionaries
    """
    return [
        {"role": "user", "content": "First message"},
        {"role": "assistant", "content": "First response"},
        {"role": "user", "content": "Second message"},
    ]


@pytest.fixture
def sample_metadata() -> MessageMetadata:
    """Sample MessageMetadata for testing.

    Returns:
        MessageMetadata instance with default values
    """
    metadata = MessageMetadata()
    metadata.turn_index = 0
    return metadata


@pytest.fixture
def sample_turn_state() -> TurnState:
    """Sample TurnState for testing transformers.

    Returns:
        TurnState with basic fields populated
    """
    return TurnState(
        user_message="Test message",
        turn_index=0,
        user_embedding=[0.1, 0.2, 0.3],
        tools=None,
        metadata={},
    )


@pytest.fixture
def sample_context_window() -> ContextWindow:
    """Sample ContextWindow with messages.

    Returns:
        ContextWindow with pre-populated messages
    """
    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!"),
    ]
    return ContextWindow(messages=messages, max_tokens=4096)


@pytest.fixture
def empty_context_window() -> ContextWindow:
    """Empty ContextWindow for testing.

    Returns:
        ContextWindow with no messages
    """
    return ContextWindow(messages=[], max_tokens=4096)


@pytest.fixture
def sample_tool_calls() -> list[dict]:
    """Sample tool calls for testing.

    Returns:
        List of tool call dictionaries
    """
    return [
        {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "San Francisco"}',
            },
        }
    ]
