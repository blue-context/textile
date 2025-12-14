"""Tests for async aembedding() API."""

from unittest.mock import AsyncMock, patch

import pytest

from textile.lite.embeddings import aembedding


@pytest.mark.parametrize("input_type", ["single", "multiple"])
async def test_aembedding_without_storage(mock_embedding_response, input_type):
    """Generate embeddings without storage."""
    test_input = "test text" if input_type == "single" else ["text1", "text2"]
    with patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_embedding_response):
        result = await aembedding(model="text-embedding-3-small", input=test_input)
        assert result == mock_embedding_response


async def test_aembedding_with_storage_configured(mock_embedding_response, mock_async_store):
    """Store embedding event when storage configured."""
    with patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_embedding_response):
        with patch("textile.lite.embeddings.get_config") as mock_config:
            mock_config.return_value.async_store = mock_async_store
            result = await aembedding(
                model="text-embedding-3-small",
                input="test",
                store_in_conversation="conv_123"
            )
            assert result == mock_embedding_response
            mock_async_store.store_embedding_event.assert_called_once()


async def test_aembedding_storage_failure_handling(mock_embedding_response, mock_async_store):
    """Embedding succeeds even if storage fails."""
    mock_async_store.store_embedding_event.side_effect = Exception("Storage error")
    with patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_embedding_response):
        with patch("textile.lite.embeddings.get_config") as mock_config:
            mock_config.return_value.async_store = mock_async_store
            result = await aembedding(
                model="text-embedding-3-small",
                input="test",
                store_in_conversation="conv_123"
            )
            assert result == mock_embedding_response


async def test_aembedding_storage_not_configured():
    """Raise error when storage requested but not configured."""
    with patch("litellm.aembedding", new_callable=AsyncMock):
        with patch("textile.lite.embeddings.get_config") as mock_config:
            mock_config.return_value.async_store = None
            with pytest.raises(RuntimeError, match="Async store not configured"):
                await aembedding(
                    model="text-embedding-3-small",
                    input="test",
                    store_in_conversation="conv_123"
                )


@pytest.mark.parametrize("dimensions,encoding_format", [
    (None, None),
    (1536, "float"),
    (512, "base64"),
])
async def test_aembedding_with_litellm_kwargs(mock_embedding_response, dimensions, encoding_format):
    """Pass additional kwargs to litellm."""
    kwargs = {}
    if dimensions:
        kwargs["dimensions"] = dimensions
    if encoding_format:
        kwargs["encoding_format"] = encoding_format

    with patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_embedding_response) as mock_llm:
        await aembedding(model="text-embedding-3-small", input="test", **kwargs)
        mock_llm.assert_called_once()
        for key, value in kwargs.items():
            assert mock_llm.call_args.kwargs[key] == value


async def test_aembedding_metadata_extraction(mock_embedding_response):
    """Extract metadata from response correctly."""
    with patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_embedding_response):
        with patch("textile.lite.embeddings.get_config") as mock_config:
            mock_config.return_value.async_store = None
            result = await aembedding(model="text-embedding-3-small", input="test")
            assert hasattr(result, "data")
            assert hasattr(result, "usage")
