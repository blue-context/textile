"""Tests for async acompletion() API."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from textile.lite.completion import acompletion


@pytest.mark.parametrize("stream", [True, False])
async def test_acompletion_without_transformers(sample_messages, mock_completion_response, stream):
    """Direct passthrough when no transformers configured."""
    with patch(
        "litellm.acompletion", new_callable=AsyncMock, return_value=mock_completion_response
    ):
        with patch("textile.lite.completion.get_config") as mock_config:
            mock_config.return_value.transformers = None
            result = await acompletion(model="gpt-4", messages=sample_messages, stream=stream)
            assert result == mock_completion_response


async def test_acompletion_with_transformers(
    sample_messages, mock_completion_response, mock_transformer
):
    """Apply transformers and return modified response."""
    with patch(
        "litellm.acompletion", new_callable=AsyncMock, return_value=mock_completion_response
    ):
        with patch("textile.lite.completion.get_config") as mock_config:
            mock_config.return_value.transformers = None
            with patch("litellm.get_max_tokens", return_value=4096):
                mock_transformer.transform.return_value = (
                    Mock(render=Mock(return_value=sample_messages)),
                    Mock(tools=None),
                )
                result = await acompletion(
                    model="gpt-4", messages=sample_messages, transformers=[mock_transformer]
                )
                assert result is not None
                mock_transformer.transform.assert_called_once()


@pytest.mark.parametrize("max_tokens", [None, 2048, 8192])
async def test_acompletion_max_tokens_handling(
    sample_messages, mock_completion_response, max_tokens
):
    """Handle max_tokens from kwargs or model metadata."""
    with patch(
        "litellm.acompletion", new_callable=AsyncMock, return_value=mock_completion_response
    ):
        with patch("textile.lite.completion.get_config") as mock_config:
            mock_config.return_value.transformers = None
            kwargs = {"max_tokens": max_tokens} if max_tokens else {}
            with patch("litellm.get_max_tokens", return_value=4096):
                result = await acompletion(model="gpt-4", messages=sample_messages, **kwargs)
                assert result is not None


async def test_acompletion_with_tools(sample_messages, mock_completion_response):
    """Pass tools to litellm."""
    tools = [{"type": "function", "function": {"name": "test"}}]
    with patch(
        "litellm.acompletion", new_callable=AsyncMock, return_value=mock_completion_response
    ) as mock_llm:
        with patch("textile.lite.completion.get_config") as mock_config:
            mock_config.return_value.transformers = None
            await acompletion(
                model="gpt-4", messages=sample_messages, tools=tools, tool_choice="auto"
            )
            mock_llm.assert_called_once()
            assert mock_llm.call_args.kwargs["tools"] == tools


async def test_acompletion_debug_trace(sample_messages, mock_completion_response, mock_transformer):
    """Attach debug trace when debug=True."""
    with patch(
        "litellm.acompletion", new_callable=AsyncMock, return_value=mock_completion_response
    ):
        with patch("textile.lite.completion.get_config") as mock_config:
            mock_config.return_value.transformers = None
            with patch("litellm.get_max_tokens", return_value=4096):
                mock_transformer.transform.return_value = (
                    Mock(render=Mock(return_value=sample_messages), messages=[], max_tokens=4096),
                    Mock(tools=None, user_message="test", metadata={}),
                )
                result = await acompletion(
                    model="gpt-4",
                    messages=sample_messages,
                    transformers=[mock_transformer],
                    debug=True,
                )
                assert hasattr(result, "_textile_trace")


async def test_acompletion_passthrough_streaming(sample_messages):
    """Streaming works when no patterns."""

    async def async_gen():
        yield Mock()

    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=async_gen()):
        with patch("textile.lite.completion.get_config") as mock_config:
            mock_config.return_value.transformers = None
            result = await acompletion(model="gpt-4", messages=sample_messages, stream=True)
            assert result is not None
