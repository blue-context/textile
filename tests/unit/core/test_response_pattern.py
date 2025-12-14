"""Concise OnPattern tests."""

import re

import pytest

from textile.core.response_pattern import OnPattern


class TestPatternCreation:
    def test_string_pattern_literal(self) -> None:
        pattern = OnPattern("<PHONE>", "555-1234")
        assert isinstance(pattern.pattern, re.Pattern) and pattern.replacement == "555-1234"

    def test_regex_pattern_compiled(self) -> None:
        regex = re.compile(r"<PHONE_(\d+)>")
        assert OnPattern(regex, "redacted").pattern == regex

    def test_ignore_case_flag(self) -> None:
        assert OnPattern("test", "replaced", ignore_case=True).pattern.flags & re.IGNORECASE

    def test_max_replacements_default(self) -> None:
        assert OnPattern("test", "replaced").max_replacements == -1

    @pytest.mark.parametrize("max_replacements", [0, 1, 5, 10])
    def test_max_replacements_set(self, max_replacements: int) -> None:
        assert OnPattern("test", "replaced", max_replacements=max_replacements).max_replacements == max_replacements

    @pytest.mark.parametrize("invalid_pattern", [123, None, [], {}])
    def test_invalid_pattern_type_raises_error(self, invalid_pattern: object) -> None:
        with pytest.raises(TypeError, match="pattern must be str or re.Pattern"):
            OnPattern(invalid_pattern, "replaced")  # type: ignore[arg-type]

    @pytest.mark.parametrize("invalid_replacement", [123, None, []])
    def test_invalid_replacement_type_raises_error(self, invalid_replacement: object) -> None:
        with pytest.raises(TypeError, match="replacement must be str or Callable"):
            OnPattern("test", invalid_replacement)  # type: ignore[arg-type]


class TestStringReplacement:
    def test_get_replacement_with_string(self) -> None:
        pattern = OnPattern("test", "replaced")
        match = re.match(r"test", "test")
        assert match is not None and pattern.get_replacement(match) == "replaced"

    def test_string_pattern_escapes_special_chars(self) -> None:
        assert OnPattern("<PHONE>", "555-1234").pattern.pattern == re.escape("<PHONE>")


class TestCallableReplacement:
    def test_callable_no_args(self) -> None:
        pattern = OnPattern("test", lambda: "dynamic")
        match = re.match(r"test", "test")
        assert match is not None and pattern.get_replacement(match) == "dynamic"

    def test_callable_with_match_arg(self) -> None:
        pattern = OnPattern(r"<PHONE_(\d+)>", lambda m: f"phone_{m.group(1)}")
        match = re.match(r"<PHONE_(\d+)>", "<PHONE_123>")
        assert match is not None and pattern.get_replacement(match) == "phone_123"

    def test_callable_return_value_converted_to_string(self) -> None:
        pattern = OnPattern("test", lambda: 42)
        match = re.match(r"test", "test")
        assert match is not None and pattern.get_replacement(match) == "42"


class TestPatternWarnings:
    def test_compiled_pattern_with_ignore_case_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        OnPattern(re.compile("test"), "replaced", ignore_case=True)
        assert "ignore_case=True but pattern already compiled" in caplog.text


class TestEdgeCases:
    def test_empty_string_pattern(self) -> None:
        assert isinstance(OnPattern("", "replaced").pattern, re.Pattern)

    def test_empty_string_replacement(self) -> None:
        assert OnPattern("test", "").replacement == ""
