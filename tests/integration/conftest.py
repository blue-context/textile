"""Shared fixtures for integration tests."""

from types import SimpleNamespace

import pytest


@pytest.fixture
def mock_litellm_response():
    """Mock LiteLLM completion response for integration tests."""
    message = SimpleNamespace(content="Mocked LLM response")
    choice = SimpleNamespace(message=message, index=0, finish_reason="stop")
    return SimpleNamespace(
        choices=[choice],
        model="gpt-4",
        usage=SimpleNamespace(prompt_tokens=50, completion_tokens=100, total_tokens=150),
    )


@pytest.fixture
def mock_litellm_streaming():
    """Mock LiteLLM streaming chunks."""
    chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="Hello "), index=0, finish_reason=None
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="world!"), index=0, finish_reason=None
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(delta=SimpleNamespace(content=None), index=0, finish_reason="stop")
            ]
        ),
    ]
    return iter(chunks)


@pytest.fixture
def mock_async_litellm_streaming():
    """Mock async LiteLLM streaming."""

    async def async_gen():
        chunks = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="Async "), index=0, finish_reason=None
                    )
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="response!"), index=0, finish_reason=None
                    )
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=None), index=0, finish_reason="stop"
                    )
                ]
            ),
        ]
        for chunk in chunks:
            yield chunk

    return async_gen()


@pytest.fixture
def conversation_messages():
    """Realistic conversation for integration tests."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
        {"role": "user", "content": "Tell me about decorators."},
    ]


@pytest.fixture
def tool_catalog():
    """Large tool catalog for tool selection tests."""
    return [
        {
            "type": "function",
            "function": {
                "name": f"tool_{i}",
                "description": f"Tool for {'weather' if i < 5 else 'database' if i < 10 else 'file'} operations",
            },
        }
        for i in range(20)
    ]


@pytest.fixture
def mock_embedding():
    """Mock embedding vector."""
    return [0.1] * 1536
