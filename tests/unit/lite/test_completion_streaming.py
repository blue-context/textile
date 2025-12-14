"""Tests for completion streaming helpers."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from textile.lite.completion import (
    _handle_streaming_response,
    _process_stream_chunk,
)


def test_process_stream_chunk_with_content():
    """Process chunk with content."""
    chunk = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="test"))])
    handler = Mock()
    handler.transform_chunk.return_value = "transformed"

    result_chunk, should_yield = _process_stream_chunk(chunk, handler)

    assert should_yield is True
    assert result_chunk.choices[0].delta.content == "transformed"


def test_process_stream_chunk_no_content():
    """Process chunk without content."""
    chunk = SimpleNamespace(choices=[])
    handler = Mock()

    result_chunk, should_yield = _process_stream_chunk(chunk, handler)

    assert should_yield is True
    assert result_chunk == chunk


def test_process_stream_chunk_empty_transformed():
    """Don't yield when transformation returns empty."""
    chunk = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="test"))])
    handler = Mock()
    handler.transform_chunk.return_value = ""

    result_chunk, should_yield = _process_stream_chunk(chunk, handler)

    assert should_yield is False


def test_handle_streaming_response_type_error():
    """Raise error when patterns is not list."""
    with pytest.raises(TypeError, match="patterns must be list"):
        _handle_streaming_response(iter([]), "not a list", is_async=False)
