"""Tests for token counting utilities."""

from unittest.mock import patch

import pytest

from textile.lite.tokens import count_tokens


@pytest.mark.parametrize("model", ["gpt-4", "gpt-3.5-turbo", "claude-3-opus"])
def test_count_tokens_with_messages(model, sample_messages):
    """Count tokens for message list."""
    with patch("textile.lite.tokens.litellm_token_counter", return_value=42):
        result = count_tokens(model=model, messages=sample_messages)
        assert result == 42


@pytest.mark.parametrize("text", ["Hello world", "This is a longer test message"])
def test_count_tokens_with_text(text):
    """Count tokens for plain text."""
    with patch("textile.lite.tokens.litellm_token_counter", return_value=10):
        result = count_tokens(model="gpt-4", text=text)
        assert result == 10


def test_count_tokens_with_tools(sample_messages):
    """Count tokens including tool definitions."""
    tools = [{"type": "function", "function": {"name": "test", "description": "test"}}]
    with patch("textile.lite.tokens.litellm_token_counter", return_value=50):
        result = count_tokens(model="gpt-4", messages=sample_messages, tools=tools)
        assert result == 50


def test_count_tokens_with_custom_tokenizer(sample_messages):
    """Use custom tokenizer if provided."""
    custom_tokenizer = object()
    with patch("textile.lite.tokens.litellm_token_counter", return_value=30):
        result = count_tokens(
            model="gpt-4", messages=sample_messages, custom_tokenizer=custom_tokenizer
        )
        assert result == 30


def test_count_tokens_fallback_heuristic_messages(sample_messages):
    """Fallback to heuristic when tokenizer fails."""
    with patch("textile.lite.tokens.litellm_token_counter", side_effect=Exception("No tokenizer")):
        result = count_tokens(model="unknown-model", messages=sample_messages)
        assert result > 0


def test_count_tokens_fallback_heuristic_text():
    """Fallback to heuristic for text."""
    with patch("textile.lite.tokens.litellm_token_counter", side_effect=Exception("No tokenizer")):
        result = count_tokens(model="unknown-model", text="Hello world")
        assert result >= 1


def test_count_tokens_empty_messages():
    """Handle empty message list."""
    with patch("textile.lite.tokens.litellm_token_counter", return_value=0):
        result = count_tokens(model="gpt-4", messages=[])
        assert result == 0


def test_count_tokens_empty_text():
    """Handle empty text string."""
    with patch("textile.lite.tokens.litellm_token_counter", return_value=0):
        result = count_tokens(model="gpt-4", text="")
        assert result == 0


def test_count_tokens_neither_messages_nor_text():
    """Raise error when neither messages nor text provided."""
    with pytest.raises(ValueError, match="Must provide either messages or text"):
        count_tokens(model="gpt-4")


def test_count_tokens_with_response_tokens(sample_messages):
    """Include response token estimate."""
    with patch("textile.lite.tokens.litellm_token_counter", return_value=60):
        result = count_tokens(model="gpt-4", messages=sample_messages, count_response_tokens=True)
        assert result == 60


def test_count_tokens_fallback_heuristic_calculation():
    """Verify fallback heuristic calculation."""
    with patch("textile.lite.tokens.litellm_token_counter", side_effect=Exception("No tokenizer")):
        text = "x" * 40
        result = count_tokens(model="unknown", text=text)
        expected = max(1, 40 // 4)
        assert result == expected


def test_count_tokens_message_overhead():
    """Verify message overhead in fallback."""
    with patch("textile.lite.tokens.litellm_token_counter", side_effect=Exception("No tokenizer")):
        messages = [{"role": "user", "content": "x"}]
        result = count_tokens(model="unknown", messages=messages)
        assert result >= 4
