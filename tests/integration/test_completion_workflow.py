"""Integration tests for basic completion workflows."""
from unittest.mock import patch
import pytest
from textile import completion
from textile.transformers import DecayTransformer

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
                    model="gpt-4", messages=conversation_messages,
                    transformers=[DecayTransformer(half_life_turns=3)])
                assert response is not None
                mock_llm.assert_called_once()
                assert "messages" in mock_llm.call_args.kwargs

@pytest.mark.parametrize("max_tokens", [None, 2048, 8192])
def test_completion_max_tokens_configuration(conversation_messages, mock_litellm_response, max_tokens):
    """Test max_tokens handling from kwargs or model metadata."""
    with patch("litellm.completion", return_value=mock_litellm_response):
        with patch("textile.lite.completion.get_config") as mock_config:
            mock_config.return_value.transformers = None
            with patch("litellm.get_max_tokens", return_value=4096):
                kwargs = {"max_tokens": max_tokens} if max_tokens else {}
                response = completion(
                    model="gpt-4", messages=conversation_messages,
                    transformers=[DecayTransformer()], **kwargs)
                assert response is not None

def test_completion_with_tools(conversation_messages, mock_litellm_response):
    """Test completion with tool definitions."""
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Get weather for location"}}]
    with patch("litellm.completion", return_value=mock_litellm_response) as mock_llm:
        with patch("textile.lite.completion.get_config") as mock_config:
            mock_config.return_value.transformers = None
            response = completion(
                model="gpt-4", messages=conversation_messages,
                tools=tools, tool_choice="auto")
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
                    model="gpt-4", messages=conversation_messages,
                    transformers=[DecayTransformer()], debug=True)
                assert response is not None
                assert hasattr(response, "_textile_trace")
                assert "context_size" in response._textile_trace
                assert "transformers" in response._textile_trace
