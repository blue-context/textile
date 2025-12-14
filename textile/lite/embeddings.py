"""Transparent wrapper around LiteLLM embedding with modern Python 3.11+ patterns."""

import asyncio
from typing import Any

import litellm

from textile.config import get_config
from textile.utils.async_helpers import run_sync


def _extract_embedding_data(
    response: Any,
    input: str | list[str],
) -> tuple[list[str], list[list[float]]]:
    """Extract input texts and embeddings from response."""
    input_texts = [input] if isinstance(input, str) else input
    embeddings = [item.embedding for item in response.data]
    return input_texts, embeddings


def _build_metadata(response: Any, dimension: int) -> dict[str, Any]:
    """Build metadata dict from response."""
    metadata: dict[str, Any] = {
        "dimension": dimension,
        "provider": "litellm",
        "object": response.object if hasattr(response, "object") else "list",
    }

    if hasattr(response, "usage") and response.usage:
        metadata["usage"] = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
            "total_tokens": getattr(response.usage, "total_tokens", 0),
        }

    return metadata


def _store_embedding_event_sync(
    store: Any,
    conversation_id: str,
    model: str,
    input_texts: list[str],
    embeddings: list[list[float]],
    metadata: dict[str, Any],
) -> None:
    """Store embedding event synchronously."""
    try:
        run_sync(store.store_embedding_event(
            conversation_id=conversation_id,
            model=model,
            input_texts=input_texts,
            embeddings=embeddings,
            metadata=metadata,
        ))
    except Exception:
        # Storage is non-critical - embedding succeeds even if storage fails.
        pass


def _execute_embedding(
    model: str,
    input: str | list[str],
    store_in_conversation: str | None,
    *,
    is_async: bool = False,
    **litellm_kwargs: Any,
) -> Any:
    """Execute embedding call with optional storage.

    Args:
        model: Model name
        input: Text to embed
        store_in_conversation: Conversation ID for tracking
        is_async: Use async variant
        **litellm_kwargs: Passed to litellm

    Returns:
        Response or coroutine if is_async

    Raises:
        RuntimeError: If storage requested but not configured
    """
    config = get_config()

    if is_async:
        response_coro = litellm.aembedding(model=model, input=input, **litellm_kwargs)
    else:
        response = litellm.embedding(model=model, input=input, **litellm_kwargs)

    if store_in_conversation is None:
        return response_coro if is_async else response
    if is_async:
        store = config.async_store
        if store is None:
            raise RuntimeError(
                "Async store not configured. For aembedding() with store_in_conversation, "
                "configure an async store:\n"
                "  textile.configure(async_storage=AsyncInMemoryStore())"
            )
    else:
        store = config._store
        if store is None:
            raise RuntimeError(
                "Sync store not configured. For embedding() with store_in_conversation, "
                "configure a store:\n"
                "  textile.configure(storage=InMemoryStore())"
            )
    if is_async:
        async def _async_embed_and_store():
            resp = await response_coro
            input_texts, embeddings = _extract_embedding_data(resp, input)
            metadata = _build_metadata(resp, len(embeddings[0]) if embeddings else 0)

            # Storage is non-critical - embedding succeeds even if storage fails.
            try:
                await store.store_embedding_event(
                    conversation_id=store_in_conversation,
                    model=model,
                    input_texts=input_texts,
                    embeddings=embeddings,
                    metadata=metadata,
                )
            except Exception:
                pass

            return resp
        return _async_embed_and_store()
    input_texts, embeddings = _extract_embedding_data(response, input)
    metadata = _build_metadata(response, len(embeddings[0]) if embeddings else 0)
    _store_embedding_event_sync(
        store, store_in_conversation, model,
        input_texts, embeddings, metadata
    )
    return response


def embedding(
    model: str,
    input: str | list[str],
    store_in_conversation: str | None = None,
    **litellm_kwargs: Any,
) -> Any:
    """Generate embeddings with optional conversation storage.

    Args:
        model: Model name
        input: Text to embed
        store_in_conversation: Conversation ID for tracking
        **litellm_kwargs: Passed to litellm

    Returns:
        EmbeddingResponse from litellm

    Examples:
        >>> response = textile.embedding(
        ...     model="text-embedding-3-small",
        ...     input="Hello world"
        ... )
        >>> response = textile.embedding(
        ...     model="text-embedding-3-small",
        ...     input=["doc1", "doc2"],
        ...     store_in_conversation="conv_123"
        ... )
    """
    return _execute_embedding(
        model, input, store_in_conversation,
        is_async=False, **litellm_kwargs
    )


async def aembedding(
    model: str,
    input: str | list[str],
    store_in_conversation: str | None = None,
    **litellm_kwargs: Any,
) -> Any:
    """Generate embeddings asynchronously with optional storage.

    Args:
        model: Model name
        input: Text to embed
        store_in_conversation: Conversation ID for tracking
        **litellm_kwargs: Passed to litellm

    Returns:
        EmbeddingResponse from litellm

    Examples:
        >>> async def main():
        ...     response = await textile.aembedding(
        ...         model="text-embedding-3-small",
        ...         input="Hello world"
        ...     )
    """
    return await _execute_embedding(
        model, input, store_in_conversation,
        is_async=True, **litellm_kwargs
    )
