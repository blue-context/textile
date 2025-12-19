"""Integration tests for streaming workflows."""

from unittest.mock import patch

import pytest

from textile import acompletion, completion
from textile.transformers.base import ContextTransformer


class MockDecayTransformer(ContextTransformer):
    """Mock transformer for testing integration workflows."""

    def __init__(self, half_life_turns: int = 5, threshold: float = 0.1):
        """Initialize mock decay transformer.

        Args:
            half_life_turns: Turns until prominence decays by half
            threshold: Prominence threshold for pruning
        """
        self.half_life_turns = half_life_turns
        self.threshold = threshold

    def transform(self, context, state):
        """Return context unchanged for integration tests."""
        return context, state


def test_streaming_completion_basic(conversation_messages, mock_litellm_streaming):
    """Test streaming response without transformers."""
    with patch("litellm.completion", return_value=mock_litellm_streaming):
        with patch("textile.lite.completion.get_config") as mock_config:
            mock_config.return_value.transformers = None

            response = completion(model="gpt-4", messages=conversation_messages, stream=True)

            chunks = list(response)
            assert len(chunks) > 0


def test_streaming_with_transformer(conversation_messages, mock_litellm_streaming):
    """Test streaming → pattern transformation → chunks."""
    with patch("litellm.completion", return_value=mock_litellm_streaming):
        with patch("textile.lite.completion.get_config") as mock_config:
            mock_config.return_value.transformers = None
            with patch("litellm.get_max_tokens", return_value=4096):
                response = completion(
                    model="gpt-4",
                    messages=conversation_messages,
                    transformers=[MockDecayTransformer()],
                    stream=True,
                )

                chunks = list(response)
                assert len(chunks) > 0


def test_streaming_collects_chunks(conversation_messages, mock_litellm_streaming):
    """Test streaming collects all chunks properly."""
    with patch("litellm.completion", return_value=mock_litellm_streaming):
        with patch("textile.lite.completion.get_config") as mock_config:
            mock_config.return_value.transformers = None
            with patch("litellm.get_max_tokens", return_value=4096):
                response = completion(
                    model="gpt-4",
                    messages=conversation_messages,
                    transformers=[MockDecayTransformer()],
                    stream=True,
                )

                collected = []
                for chunk in response:
                    if hasattr(chunk, "choices") and chunk.choices:
                        if hasattr(chunk.choices[0], "delta"):
                            content = getattr(chunk.choices[0].delta, "content", None)
                            if content:
                                collected.append(content)

                # Verify chunks collected
                assert len(collected) > 0


@pytest.mark.asyncio
async def test_async_streaming_workflow(conversation_messages, mock_async_litellm_streaming):
    """Test async streaming → transformers → chunks."""
    with patch("litellm.acompletion", return_value=mock_async_litellm_streaming):
        with patch("textile.lite.completion.get_config") as mock_config:
            mock_config.return_value.transformers = None
            with patch("litellm.get_max_tokens", return_value=4096):
                response = await acompletion(
                    model="gpt-4",
                    messages=conversation_messages,
                    transformers=[MockDecayTransformer()],
                    stream=True,
                )

                chunks = []
                async for chunk in response:
                    chunks.append(chunk)

                assert len(chunks) > 0
