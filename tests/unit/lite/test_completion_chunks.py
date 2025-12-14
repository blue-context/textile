"""Tests for completion chunk processing."""

from types import SimpleNamespace

import pytest

from textile.lite.completion import (
    _create_flush_chunk,
    _extract_chunk_content,
)


@pytest.mark.parametrize(
    "has_choices,has_delta,has_content,expected",
    [
        (False, False, False, None),
        (True, False, False, None),
        (True, True, False, None),
        (True, True, True, "content"),
    ],
)
def test_extract_chunk_content(has_choices, has_delta, has_content, expected):
    """Extract content from streaming chunks."""
    if not has_choices:
        chunk = SimpleNamespace()
        assert _extract_chunk_content(chunk) is None
    elif not has_delta:
        chunk = SimpleNamespace(choices=[SimpleNamespace()])
        assert _extract_chunk_content(chunk) is None
    elif not has_content:
        chunk = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace())])
        assert _extract_chunk_content(chunk) is None
    else:
        chunk = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="content"))])
        assert _extract_chunk_content(chunk) == "content"


def test_create_flush_chunk():
    """Create final chunk for flushed content."""
    chunk = _create_flush_chunk("final content")
    assert chunk is not None
    assert chunk.choices[0].delta.content == "final content"
    assert chunk.choices[0].finish_reason is None
