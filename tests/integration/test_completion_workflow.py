"""Integration tests for basic completion workflows."""

from unittest.mock import patch

import pytest

from textile import completion
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


def test_basic_completion_no_transformers(conversation_messages, mock_litellm_response):
    """Test messages → LLM without transformers."""
    with patch("litellm.completion", return_value=mock_litellm_response):
        with patch("textile.lite.completion.get_config") as mock_config:
            mock_config.return_value.transformers = None

            response = completion(model="gpt-4", messages=conversation_messages)
            assert response is not None
            assert response.choices[0].message.content == "Mocked LLM response"


def test_completion_with_single_transformer(conversation_messages, mock_litellm_response):
    """Test messages → decay transformer → LLM."""
    with patch("litellm.completion", return_value=mock_litellm_response) as mock_llm:
        with patch("textile.lite.completion.get_config") as mock_config:
            mock_config.return_value.transformers = None
            with patch("litellm.get_max_tokens", return_value=4096):
                response = completion(
                    model="gpt-4",
                    messages=conversation_messages,
                    transformers=[MockDecayTransformer(half_life_turns=3)],
                )
                assert response is not None
                mock_llm.assert_called_once()
                assert "messages" in mock_llm.call_args.kwargs


@pytest.mark.parametrize("max_tokens", [None, 2048, 8192])
def test_completion_max_tokens_configuration(
    conversation_messages, mock_litellm_response, max_tokens
):
    """Test max_tokens handling from kwargs or model metadata."""
    with patch("litellm.completion", return_value=mock_litellm_response):
        with patch("textile.lite.completion.get_config") as mock_config:
            mock_config.return_value.transformers = None
            with patch("litellm.get_max_tokens", return_value=4096):
                kwargs = {"max_tokens": max_tokens} if max_tokens else {}
                response = completion(
                    model="gpt-4",
                    messages=conversation_messages,
                    transformers=[MockDecayTransformer()],
                    **kwargs,
                )
                assert response is not None


def test_completion_with_tools(conversation_messages, mock_litellm_response):
    """Test completion with tool definitions."""
    tools = [
        {
            "type": "function",
            "function": {"name": "get_weather", "description": "Get weather for location"},
        }
    ]
    with patch("litellm.completion", return_value=mock_litellm_response) as mock_llm:
        with patch("textile.lite.completion.get_config") as mock_config:
            mock_config.return_value.transformers = None
            response = completion(
                model="gpt-4", messages=conversation_messages, tools=tools, tool_choice="auto"
            )
            assert response is not None
            call_kwargs = mock_llm.call_args.kwargs
            assert call_kwargs["tools"] == tools
            assert call_kwargs["tool_choice"] == "auto"


def test_completion_debug_mode(conversation_messages, mock_litellm_response):
    """Test debug mode attaches trace to response."""
    with patch("litellm.completion", return_value=mock_litellm_response):
        with patch("textile.lite.completion.get_config") as mock_config:
            mock_config.return_value.transformers = None
            with patch("litellm.get_max_tokens", return_value=4096):
                response = completion(
                    model="gpt-4",
                    messages=conversation_messages,
                    transformers=[MockDecayTransformer()],
                    debug=True,
                )
                assert response is not None
                assert hasattr(response, "_textile_trace")
                assert "context_size" in response._textile_trace
                assert "transformers" in response._textile_trace
