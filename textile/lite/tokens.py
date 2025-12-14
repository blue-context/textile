"""Transparent wrapper around LiteLLM token counting with modern Python 3.11+ patterns."""

import logging
from typing import Any

from litellm import (
    create_pretrained_tokenizer,
    create_tokenizer,
    decode,
    encode,
    token_counter as litellm_token_counter,
)

logger = logging.getLogger(__name__)

# OpenAI empirical observation: ~4 chars ≈ 1 token for English
CHARS_PER_TOKEN_HEURISTIC: int = 4

# Role markers and API structure overhead per message
MESSAGE_OVERHEAD_TOKENS: int = 4


def count_tokens(
    model: str,
    messages: list[dict[str, Any]] | None = None,
    text: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    custom_tokenizer: Any | None = None,
    count_response_tokens: bool = False,
) -> int:
    """Count tokens using model-specific tokenizer.

    Falls back to heuristic if tokenization fails.

    Args:
        model: Model name
        messages: Message dicts
        text: Plain text alternative
        tools: Tool definitions
        tool_choice: Tool selection
        custom_tokenizer: Custom tokenizer instance
        count_response_tokens: Include response estimate

    Raises:
        ValueError: If neither messages nor text provided
    """
    # Empty strings/lists are valid, but at least one must be provided
    if messages is None and text is None:
        raise ValueError("Must provide either messages or text parameter")

    try:
        return litellm_token_counter(
            model=model,
            messages=messages,
            text=text,
            tools=tools,  # type: ignore[arg-type]
            tool_choice=tool_choice,  # type: ignore[arg-type]
            custom_tokenizer=custom_tokenizer,
            count_response_tokens=count_response_tokens,
        )
    except Exception as e:
        # Prevents hard failures when model tokenizer unknown
        logger.warning(
            f"Token counting failed for model '{model}': {e}. Using fallback heuristic."
        )
        return _fallback_token_count(messages, text)


def _fallback_token_count(
    messages: list[dict[str, Any]] | None,
    text: str | None,
) -> int:
    """Estimate tokens using character heuristic."""
    if text:
        return max(1, len(text) // CHARS_PER_TOKEN_HEURISTIC)

    if messages:
        total = 0
        for msg in messages:
            if "content" in msg and isinstance(msg["content"], str):
                total += max(1, len(msg["content"]) // CHARS_PER_TOKEN_HEURISTIC)
            total += MESSAGE_OVERHEAD_TOKENS
        return total

    return 0


# Re-export LiteLLM utilities for drop-in replacement
__all__ = [
    "count_tokens",
    "create_pretrained_tokenizer",
    "create_tokenizer",
    "encode",
    "decode",
]
