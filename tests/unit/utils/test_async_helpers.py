"""Tests for async/sync compatibility helpers."""

import asyncio

import pytest

from textile.utils.async_helpers import run_sync


class TestRunSync:
    """Test run_sync helper function."""

    def test_runs_coroutine_in_sync_context(self, sample_coroutine):
        result = run_sync(sample_coroutine(5))
        assert result == 10

    @pytest.mark.parametrize("value,expected", [
        (1, 2), (10, 20), (100, 200), (0, 0)
    ])
    def test_returns_coroutine_result(self, sample_coroutine, value, expected):
        result = run_sync(sample_coroutine(value))
        assert result == expected

    def test_propagates_exceptions(self, failing_coroutine):
        with pytest.raises(ValueError, match="Test error"):
            run_sync(failing_coroutine())

    @pytest.mark.asyncio
    async def test_raises_error_from_async_context(self, sample_coroutine):
        """Verify helpful error when called from async context."""
        with pytest.raises(RuntimeError, match="Cannot call sync API from async context"):
            run_sync(sample_coroutine(5))

    @pytest.mark.asyncio
    async def test_error_message_suggests_async_variants(self, sample_coroutine):
        """Verify error message includes helpful guidance."""
        with pytest.raises(RuntimeError) as exc_info:
            run_sync(sample_coroutine(5))

        error_msg = str(exc_info.value)
        assert "acompletion()" in error_msg
        assert "aembedding()" in error_msg
        assert "await" in error_msg

    def test_handles_immediate_return_coroutine(self):
        """Test coroutine that returns immediately without await."""
        async def immediate():
            return 42

        result = run_sync(immediate())
        assert result == 42

    def test_handles_nested_awaits(self):
        """Test coroutine with nested async calls."""
        async def inner():
            await asyncio.sleep(0.001)
            return "inner"

        async def outer():
            result = await inner()
            return f"outer-{result}"

        result = run_sync(outer())
        assert result == "outer-inner"

    def test_preserves_coroutine_return_type(self):
        """Verify return type is preserved."""
        async def return_dict():
            return {"key": "value"}

        result = run_sync(return_dict())
        assert isinstance(result, dict)
        assert result == {"key": "value"}

    @pytest.mark.parametrize("return_value", [
        42, "string", [1, 2, 3], {"a": 1}, None, True
    ])
    def test_handles_various_return_types(self, return_value):
        async def return_value_coro():
            return return_value

        result = run_sync(return_value_coro())
        assert result == return_value

    def test_cleanup_after_exception(self):
        """Verify proper cleanup even when coroutine raises."""
        async def cleanup_test():
            try:
                raise ValueError("cleanup test")
            finally:
                # Ensure cleanup happens
                pass

        with pytest.raises(ValueError, match="cleanup test"):
            run_sync(cleanup_test())
