"""Integration tests for streaming workflows."""

from unittest.mock import Mock, patch

import pytest

from textile import acompletion, completion
from textile.transformers import DecayTransformer


def test_streaming_completion_basic(conversation_messages, mock_litellm_streaming):
    """Test streaming response without transformers."""
    with patch("litellm.completion", return_value=mock_litellm_streaming):
        with patch("textile.lite.completion.get_config") as mock_config:
            mock_config.return_value.transformers = None

            response = completion(
                model="gpt-4",
                messages=conversation_messages,
                stream=True
            )

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
                    transformers=[DecayTransformer()],
                    stream=True
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
                    transformers=[DecayTransformer()],
                    stream=True
                )

                collected = []
                for chunk in response:
                    if hasattr(chunk, 'choices') and chunk.choices:
                        if hasattr(chunk.choices[0], 'delta'):
                            content = getattr(chunk.choices[0].delta, 'content', None)
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
                    transformers=[DecayTransformer()],
                    stream=True
                )

                chunks = []
                async for chunk in response:
                    chunks.append(chunk)

                assert len(chunks) > 0
