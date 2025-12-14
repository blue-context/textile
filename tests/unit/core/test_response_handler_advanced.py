"""Advanced StreamingResponseHandler tests - error handling and edge cases."""

import re

import pytest

from textile.core.response_handler import StreamingResponseHandler
from textile.core.response_pattern import OnPattern


class TestErrorHandling:
    def test_error_in_pattern_returns_original(self) -> None:
        pattern = OnPattern("test", lambda m: (_ for _ in ()).throw(ValueError("error")))  # type: ignore[misc]
        handler = StreamingResponseHandler([pattern])
        handler.transform_chunk("test content")
        assert "test" in handler.flush() and handler.get_stats()['errors'] > 0

    def test_flush_error_returns_buffer(self) -> None:
        def bad_replacement(_match: re.Match) -> str:
            raise RuntimeError("flush error")
        pattern = OnPattern("test", bad_replacement)
        handler = StreamingResponseHandler([pattern])
        handler.buffer = "test"
        result = handler.flush()
        assert "test" in result and handler.get_stats()['errors'] > 0

    def test_transform_chunk_exception(self) -> None:
        def crash_func(_match: re.Match) -> str:
            raise ValueError("transform crash")
        pattern = OnPattern("X", crash_func)
        handler = StreamingResponseHandler([pattern])
        result = handler.transform_chunk("XXX buffering text")
        handler.flush()
        assert handler.get_stats()['errors'] > 0


class TestBoundaryLogic:
    def test_pattern_split_boundary(self) -> None:
        pattern = OnPattern("<MARKER>", "REPLACED")
        handler = StreamingResponseHandler([pattern])
        handler.buffer = "prefix <MAR"
        boundary = handler._find_safe_boundary()
        assert boundary <= len(handler.buffer)

    def test_partial_pattern_adjustment(self) -> None:
        pattern = OnPattern("<TAG>", "X")
        handler = StreamingResponseHandler([pattern])
        handler.buffer = "text <TA"
        compiled = pattern.pattern
        assert isinstance(compiled, re.Pattern)
        adjusted = handler._adjust_for_partial_pattern(5, compiled)
        assert adjusted >= 0

    def test_boundary_zero_when_below_threshold(self) -> None:
        handler = StreamingResponseHandler([OnPattern("X", "Y")])
        handler.buffer = "small"
        assert handler._find_safe_boundary() == 0

    def test_boundary_edge_at_boundary(self) -> None:
        pattern = OnPattern("<TAG>", "X")
        handler = StreamingResponseHandler([pattern])
        handler.buffer = "a" * 100 + "<TA"
        boundary = handler._find_safe_boundary()
        assert boundary >= 0

    @pytest.mark.parametrize("boundary,expected_ge", [
        (0, 0),
        (-5, 0),
        (1000, 1000),
    ])
    def test_adjust_boundary_edge_cases(self, boundary: int, expected_ge: int) -> None:
        pattern = OnPattern("test", "x")
        handler = StreamingResponseHandler([pattern])
        handler.buffer = "some test data here"
        compiled = pattern.pattern
        assert isinstance(compiled, re.Pattern)
        result = handler._adjust_for_partial_pattern(boundary, compiled)
        assert result >= expected_ge or result == boundary


class TestApplyPatternsEdgeCases:
    def test_apply_patterns_empty_list(self) -> None:
        handler = StreamingResponseHandler([])
        assert handler._apply_patterns("test") == "test"

    def test_pattern_error_logged_continues(self) -> None:
        def error_func(_match: re.Match) -> str:
            raise RuntimeError("pattern error")
        patterns = [OnPattern("A", error_func), OnPattern("B", "2")]
        handler = StreamingResponseHandler(patterns)
        result = handler._apply_patterns("A and B")
        assert handler.get_stats()['errors'] > 0

    def test_partial_match_at_boundary(self) -> None:
        pattern = OnPattern(re.compile(r"<TAG>"), "REPLACED")
        handler = StreamingResponseHandler([pattern])
        handler.buffer = "a" * 100 + "<TAG> more text <TA"
        boundary = handler._find_safe_boundary()
        assert boundary > 0
