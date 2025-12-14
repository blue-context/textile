"""Tests for sync embedding() API."""

from unittest.mock import Mock, patch

import pytest

from textile.lite.embeddings import embedding


@pytest.mark.parametrize("input_type", ["single", "multiple"])
def test_embedding_without_storage(mock_embedding_response, input_type):
    """Generate embeddings without storage."""
    test_input = "test text" if input_type == "single" else ["text1", "text2"]
    with patch("litellm.embedding", return_value=mock_embedding_response):
        result = embedding(model="text-embedding-3-small", input=test_input)
        assert result == mock_embedding_response


def test_embedding_with_storage_configured(mock_embedding_response, mock_sync_store):
    """Store embedding event when storage configured."""
    with patch("litellm.embedding", return_value=mock_embedding_response):
        with patch("textile.lite.embeddings.get_config") as mock_config:
            mock_config.return_value._store = mock_sync_store
            with patch("textile.lite.embeddings.run_sync") as mock_run_sync:
                result = embedding(
                    model="text-embedding-3-small",
                    input="test",
                    store_in_conversation="conv_123"
                )
                assert result == mock_embedding_response
                mock_run_sync.assert_called_once()


def test_embedding_storage_failure_handling(mock_embedding_response, mock_sync_store):
    """Embedding succeeds even if storage fails."""
    mock_sync_store.store_embedding_event.side_effect = Exception("Storage error")
    with patch("litellm.embedding", return_value=mock_embedding_response):
        with patch("textile.lite.embeddings.get_config") as mock_config:
            mock_config.return_value._store = mock_sync_store
            with patch("textile.lite.embeddings.run_sync", side_effect=Exception("Storage error")):
                result = embedding(
                    model="text-embedding-3-small",
                    input="test",
                    store_in_conversation="conv_123"
                )
                assert result == mock_embedding_response


def test_embedding_storage_not_configured():
    """Raise error when storage requested but not configured."""
    with patch("litellm.embedding"):
        with patch("textile.lite.embeddings.get_config") as mock_config:
            mock_config.return_value._store = None
            with pytest.raises(RuntimeError, match="Sync store not configured"):
                embedding(
                    model="text-embedding-3-small",
                    input="test",
                    store_in_conversation="conv_123"
                )


@pytest.mark.parametrize("dimensions,encoding_format", [
    (None, None),
    (1536, "float"),
    (512, "base64"),
])
def test_embedding_with_litellm_kwargs(mock_embedding_response, dimensions, encoding_format):
    """Pass additional kwargs to litellm."""
    kwargs = {}
    if dimensions:
        kwargs["dimensions"] = dimensions
    if encoding_format:
        kwargs["encoding_format"] = encoding_format

    with patch("litellm.embedding", return_value=mock_embedding_response) as mock_llm:
        embedding(model="text-embedding-3-small", input="test", **kwargs)
        mock_llm.assert_called_once()
        for key, value in kwargs.items():
            assert mock_llm.call_args.kwargs[key] == value


def test_embedding_metadata_extraction(mock_embedding_response):
    """Extract metadata from response correctly."""
    with patch("litellm.embedding", return_value=mock_embedding_response):
        with patch("textile.lite.embeddings.get_config") as mock_config:
            mock_config.return_value._store = None
            result = embedding(model="text-embedding-3-small", input="test")
            assert hasattr(result, "data")
            assert hasattr(result, "usage")
