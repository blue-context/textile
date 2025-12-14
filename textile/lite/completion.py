"""Transparent wrapper around LiteLLM completion with modern Python 3.11+ patterns."""

import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any

import litellm

from textile.config import get_config
from textile.core.context_window import ContextWindow
from textile.core.message import Message
from textile.core.response_handler import StreamingResponseHandler
from textile.core.turn_state import TurnState


logger = logging.getLogger(__name__)


def _build_trace(
    context: ContextWindow,
    state: TurnState,
    transformers: list,
) -> dict[str, Any]:
    """Build debug trace from transformation."""
    return {
        "context_size": len(context.messages),
        "max_tokens": context.max_tokens,
        "user_message": state.user_message,
        "transformers": [t.__class__.__name__ for t in transformers],
        "metadata": state.metadata,
    }


def _extract_chunk_content(chunk: Any) -> str | None:
    """Extract content from streaming chunk."""
    if not hasattr(chunk, 'choices') or not chunk.choices:
        return None
    if not hasattr(chunk.choices[0], 'delta'):
        return None
    if not (delta := chunk.choices[0].delta):
        return None
    return getattr(delta, 'content', None)


def _create_flush_chunk(content: str) -> Any:
    """Create final chunk for flushed content."""
    from types import SimpleNamespace

    try:
        delta = SimpleNamespace(content=content)
        choice = SimpleNamespace(delta=delta, index=0, finish_reason=None)
        return SimpleNamespace(choices=[choice])
    except Exception:
        return None


def _process_stream_chunk(
    chunk: Any,
    handler: StreamingResponseHandler,
) -> tuple[Any, bool]:
    """Process single stream chunk.

    Returns:
        (modified_chunk, should_yield)
    """
    content = _extract_chunk_content(chunk)
    if content is None:
        return chunk, True

    transformed_content = handler.transform_chunk(content)
    if not transformed_content:
        return chunk, False

    chunk.choices[0].delta.content = transformed_content
    return chunk, True


def _sync_stream_gen(response_stream: Iterator, handler: StreamingResponseHandler):
    """Sync generator for streaming response."""
    try:
        for chunk in response_stream:
            processed_chunk, should_yield = _process_stream_chunk(chunk, handler)
            if should_yield:
                yield processed_chunk
    finally:
        final_content = handler.flush()
        if final_content:
            if final_chunk := _create_flush_chunk(final_content):
                yield final_chunk


async def _async_stream_gen(response_stream: AsyncIterator, handler: StreamingResponseHandler):
    """Async generator for streaming response."""
    try:
        async for chunk in response_stream:
            processed_chunk, should_yield = _process_stream_chunk(chunk, handler)
            if should_yield:
                yield processed_chunk
    finally:
        final_content = handler.flush()
        if final_content:
            if final_chunk := _create_flush_chunk(final_content):
                yield final_chunk


def _handle_streaming_response(
    response_stream: Iterator | AsyncIterator,
    patterns: list,
    *,
    is_async: bool = False,
) -> Iterator | AsyncIterator:
    """Wrap streaming response with pattern transformation."""
    if not isinstance(patterns, list):
        raise TypeError(f"patterns must be list, got {type(patterns)}")

    handler = StreamingResponseHandler(patterns)
    if is_async:
        return _async_stream_gen(response_stream, handler)
    else:
        return _sync_stream_gen(response_stream, handler)


def _apply_response_patterns(response: Any, patterns: list) -> Any:
    """Apply patterns to non-streaming response."""
    response_choices = getattr(response, "choices", None)
    if not response_choices:
        return response

    choice = response_choices[0]
    msg = getattr(choice, "message", None)
    content = getattr(msg, "content", None) if msg else None

    if not content:
        return response

    handler = StreamingResponseHandler(patterns)
    response.choices[0].message.content = handler.transform_chunk(content) + handler.flush()
    return response


def _get_max_tokens(model: str, litellm_kwargs: dict) -> int:
    """Get max tokens from kwargs or model metadata."""
    if (max_tokens := litellm_kwargs.get("max_tokens")) is not None:
        return max_tokens

    try:
        return litellm.get_max_tokens(model)
    except Exception as e:
        logger.warning(
            f"Could not determine max_tokens for model '{model}': {e}. "
            f"Using fallback of 16384 tokens."
        )
        return 16384


def _prepare_context(
    model: str,
    messages: list[dict],
    litellm_kwargs: dict,
    tools: list | None,
) -> tuple[ContextWindow, TurnState]:
    """Prepare context window and turn state from messages."""
    max_tokens = _get_max_tokens(model, litellm_kwargs)
    message_objects = [Message.from_dict(msg) for msg in messages]

    # Assign turn indices based on conversation position
    for i, msg in enumerate(message_objects):
        msg.turn_index = i

    context = ContextWindow(messages=message_objects, max_tokens=max_tokens)

    # Current turn is the last message index
    current_turn = len(message_objects) - 1

    state = TurnState(
        user_message=messages[-1]["content"],
        turn_index=current_turn,
        tools=tools,
        metadata={}
    )

    return context, state


def _apply_transformers(
    context: ContextWindow,
    state: TurnState,
    transformers: list,
) -> tuple[ContextWindow, TurnState]:
    """Apply transformers to context and state."""
    for transformer in transformers:
        if hasattr(transformer, 'should_apply'):
            if not transformer.should_apply(context, state):
                continue
        context, state = transformer.transform(context, state)
    return context, state


def _collect_response_patterns(transformers: list, state: TurnState) -> list:
    """Collect response patterns from transformers."""
    patterns = []
    for transformer in reversed(transformers):
        if hasattr(transformer, 'on_response'):
            if transformer_patterns := transformer.on_response(state):
                patterns.extend(transformer_patterns)
    return patterns


def completion(
    model: str,
    messages: list[dict],
    transformers: list | None = None,
    tools: list | None = None,
    tool_choice: str | dict | None = None,
    debug: bool = False,
    **litellm_kwargs,
) -> Any:
    """Apply transformers and call LiteLLM.

    Args:
        model: Model name (e.g., "gpt-4")
        messages: Message dicts with 'role' and 'content'
        transformers: Optional context transformers
        tools: Tool definitions
        tool_choice: Tool choice directive
        debug: Attach _textile_trace to response
        **litellm_kwargs: stream, max_tokens, etc.

    Returns:
        LiteLLM response or stream
    """
    config = get_config()

    if not transformers and not config.transformers:
        return litellm.completion(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            **litellm_kwargs
        )

    context, state = _prepare_context(model, messages, litellm_kwargs, tools)

    transformer_list = transformers or config.transformers
    context, state = _apply_transformers(context, state, transformer_list)
    patterns = _collect_response_patterns(transformer_list, state)

    response = litellm.completion(
        model=model,
        messages=context.render(),
        tools=state.tools,
        tool_choice=tool_choice,
        **litellm_kwargs
    )

    is_streaming = litellm_kwargs.get('stream', False)
    if patterns and is_streaming:
        response = _handle_streaming_response(response, patterns, is_async=False)
    elif patterns:
        response = _apply_response_patterns(response, patterns)

    if debug and not is_streaming:
        response._textile_trace = _build_trace(context, state, transformer_list)

    return response


async def acompletion(
    model: str,
    messages: list[dict],
    transformers: list | None = None,
    tools: list | None = None,
    tool_choice: str | dict | None = None,
    debug: bool = False,
    **litellm_kwargs,
) -> Any:
    """Apply transformers and call LiteLLM (async).

    Args:
        model: Model name (e.g., "gpt-4")
        messages: Message dicts with 'role' and 'content'
        transformers: Optional context transformers
        tools: Tool definitions
        tool_choice: Tool choice directive
        debug: Attach _textile_trace to response
        **litellm_kwargs: stream, max_tokens, etc.

    Returns:
        LiteLLM response or stream
    """
    config = get_config()

    if not transformers and not config.transformers:
        return await litellm.acompletion(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            **litellm_kwargs
        )

    context, state = _prepare_context(model, messages, litellm_kwargs, tools)

    transformer_list = transformers or config.transformers
    context, state = _apply_transformers(context, state, transformer_list)
    patterns = _collect_response_patterns(transformer_list, state)

    response = await litellm.acompletion(
        model=model,
        messages=context.render(),
        tools=state.tools,
        tool_choice=tool_choice,
        **litellm_kwargs
    )

    is_streaming = litellm_kwargs.get('stream', False)
    if patterns and is_streaming:
        response = _handle_streaming_response(response, patterns, is_async=True)
    elif patterns:
        response = _apply_response_patterns(response, patterns)

    if debug and not is_streaming:
        response._textile_trace = _build_trace(context, state, transformer_list)

    return response
