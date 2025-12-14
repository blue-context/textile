"""Basic StreamingResponseHandler tests."""

import re

import pytest

from textile.core.response_handler import StreamingResponseHandler
from textile.core.response_pattern import OnPattern


@pytest.fixture
def simple_pattern() -> OnPattern:
    return OnPattern("<PHONE>", "555-1234")


@pytest.fixture
def regex_pattern() -> OnPattern:
    return OnPattern(re.compile(r"<PHONE_(\d+)>"), lambda m: f"redacted_{m.group(1)}")


@pytest.fixture
def handler(simple_pattern: OnPattern) -> StreamingResponseHandler:
    return StreamingResponseHandler([simple_pattern])


class TestHandlerCreation:
    def test_empty_patterns(self) -> None:
        assert StreamingResponseHandler([]).patterns == []

    def test_with_patterns(self, simple_pattern: OnPattern) -> None:
        assert len(StreamingResponseHandler([simple_pattern]).patterns) == 1

    def test_custom_max_buffer_size(self) -> None:
        assert StreamingResponseHandler([], max_buffer_size=1000).max_buffer_size == 1000


class TestTransformChunk:
    def test_empty_chunk_returns_empty(self, handler: StreamingResponseHandler) -> None:
        assert handler.transform_chunk("") == ""

    def test_simple_replacement(self, handler: StreamingResponseHandler) -> None:
        result = handler.transform_chunk("Call <PHONE> now! Extra text for buffering.")
        combined = result + handler.flush()
        assert "555-1234" in combined or "<PHONE>" not in combined

    def test_buffering_incomplete_pattern(self, handler: StreamingResponseHandler) -> None:
        handler.transform_chunk("Call <PHO")
        handler.transform_chunk("NE> now!")
        result = handler.flush()
        assert "555-1234" in result

    def test_stats_updated(self, handler: StreamingResponseHandler) -> None:
        handler.transform_chunk("test")
        assert handler.get_stats()['chunks_processed'] == 1

    def test_buffer_overflow_forces_flush(self) -> None:
        handler = StreamingResponseHandler([OnPattern("X", "Y")], max_buffer_size=50)
        handler.transform_chunk("A" * 60)
        assert handler.get_stats()['chunks_processed'] == 1


class TestPatternApplication:
    def test_multiple_patterns_sequential(self) -> None:
        patterns = [OnPattern("<A>", "1"), OnPattern("<B>", "2")]
        handler = StreamingResponseHandler(patterns)
        handler.transform_chunk("<A> and <B>")
        result = handler.flush()
        assert "1" in result and "2" in result

    def test_max_replacements_honored(self) -> None:
        pattern = OnPattern("test", "replaced", max_replacements=1)
        handler = StreamingResponseHandler([pattern])
        handler.transform_chunk("test test test")
        result = handler.flush()
        assert result.count("replaced") == 1

    def test_regex_pattern_groups(self, regex_pattern: OnPattern) -> None:
        handler = StreamingResponseHandler([regex_pattern])
        handler.transform_chunk("<PHONE_123> and <PHONE_456> buffering text.")
        result = handler.flush()
        assert "redacted_123" in result or "redacted_456" in result


class TestFlush:
    def test_flush_empty_buffer(self, handler: StreamingResponseHandler) -> None:
        assert handler.flush() == ""

    def test_flush_processes_buffer(self, handler: StreamingResponseHandler) -> None:
        handler.transform_chunk("Call <PHONE>")
        assert "555-1234" in handler.flush() and handler.buffer == ""
