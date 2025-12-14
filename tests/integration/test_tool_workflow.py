"""Integration tests for tool selection workflows."""
from unittest.mock import patch

from textile import completion
from textile.transformers import SemanticToolSelectionTransformer


def test_tool_selection_workflow_basic(conversation_messages, mock_litellm_response):
    """Test completion with tools passed through."""
    tools = [
        {"type": "function", "function": {"name": "get_weather", "description": "Get weather"}},
        {"type": "function", "function": {"name": "send_email", "description": "Send email"}}]
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

def test_tool_selection_transformer_initialization():
    """Test tool selection transformer initializes correctly."""
    transformer = SemanticToolSelectionTransformer(max_tools=5)
    assert transformer.max_tools == 5
    assert transformer.threshold == 0.2
    transformer = SemanticToolSelectionTransformer(max_tools=10, similarity_threshold=0.5)
    assert transformer.max_tools == 10
    assert transformer.threshold == 0.5

def test_tool_selection_with_transformer(conversation_messages, mock_litellm_response):
    """Test completion with tool selection transformer."""
    small_catalog = [
        {"type": "function", "function": {"name": "tool_1", "description": "Test tool"}},
        {"type": "function", "function": {"name": "tool_2", "description": "Another tool"}}]
    transformer = SemanticToolSelectionTransformer(max_tools=10)
    with patch("litellm.completion", return_value=mock_litellm_response) as mock_llm:
        with patch("textile.lite.completion.get_config") as mock_config:
            mock_config.return_value.transformers = None
            with patch("litellm.get_max_tokens", return_value=4096):
                response = completion(
                    model="gpt-4", messages=conversation_messages,
                    tools=small_catalog, transformers=[transformer])
                assert response is not None
                assert mock_llm.call_args.kwargs["tools"] == small_catalog

def test_tool_workflow_end_to_end(conversation_messages, mock_litellm_response):
    """Test complete workflow: messages → tools → LLM response."""
    tools = [{"type": "function", "function": {
        "name": "calculate", "description": "Perform calculations"}}]
    with patch("litellm.completion", return_value=mock_litellm_response):
        with patch("textile.lite.completion.get_config") as mock_config:
            mock_config.return_value.transformers = None
            response = completion(
                model="gpt-4", messages=conversation_messages, tools=tools)
            assert response is not None
            assert response.choices[0].message.content == "Mocked LLM response"
