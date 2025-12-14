"""Tests for completion utility functions."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

from textile.lite.completion import (
    _apply_response_patterns,
    _build_trace,
    _get_max_tokens,
)


def test_build_trace():
    """Build debug trace from context and state."""
    context = Mock(messages=[Mock(), Mock()], max_tokens=4096)
    state = Mock(user_message="test", metadata={"key": "value"})
    transformers = [Mock(__class__=Mock(__name__="TestTransformer"))]

    trace = _build_trace(context, state, transformers)

    assert trace["context_size"] == 2
    assert trace["max_tokens"] == 4096
    assert trace["user_message"] == "test"
    assert trace["transformers"] == ["TestTransformer"]
    assert trace["metadata"] == {"key": "value"}


def test_get_max_tokens_from_kwargs():
    """Get max_tokens from litellm_kwargs."""
    result = _get_max_tokens("gpt-4", {"max_tokens": 2048})
    assert result == 2048


def test_get_max_tokens_from_model():
    """Get max_tokens from model metadata."""
    with patch("litellm.get_max_tokens", return_value=8192):
        result = _get_max_tokens("gpt-4", {})
        assert result == 8192


def test_get_max_tokens_fallback():
    """Fallback to 16384 when model unknown."""
    with patch("litellm.get_max_tokens", side_effect=Exception("Unknown model")):
        result = _get_max_tokens("unknown-model", {})
        assert result == 16384


def test_apply_response_patterns():
    """Apply patterns to non-streaming response."""
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="test content"))]
    )
    patterns = [Mock()]

    with patch("textile.lite.completion.StreamingResponseHandler") as mock_handler_class:
        mock_handler = Mock()
        mock_handler.transform_chunk.return_value = "transformed"
        mock_handler.flush.return_value = ""
        mock_handler_class.return_value = mock_handler

        result = _apply_response_patterns(response, patterns)

        assert result.choices[0].message.content == "transformed"


def test_apply_response_patterns_no_content():
    """Handle response without content."""
    response = SimpleNamespace(choices=[])
    patterns = []
    result = _apply_response_patterns(response, patterns)
    assert result == response
