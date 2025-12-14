"""Tests for completion context preparation."""

from unittest.mock import Mock, patch

from textile.lite.completion import (
    _apply_transformers,
    _collect_response_patterns,
    _prepare_context,
)


def test_prepare_context(sample_messages):
    """Prepare context window and turn state."""
    with patch("litellm.get_max_tokens", return_value=4096):
        context, state = _prepare_context("gpt-4", sample_messages, {}, None)
        assert context.max_tokens == 4096
        assert len(context.messages) == 2
        assert state.user_message == "Hello"
        assert state.tools is None


def test_prepare_context_with_tools(sample_messages):
    """Include tools in turn state."""
    tools = [{"type": "function", "function": {"name": "test"}}]
    with patch("litellm.get_max_tokens", return_value=4096):
        context, state = _prepare_context("gpt-4", sample_messages, {}, tools)
        assert state.tools == tools


def test_apply_transformers():
    """Apply transformers to context and state."""
    context = Mock()
    state = Mock()
    transformer1 = Mock()
    transformer1.transform.return_value = (context, state)
    transformer2 = Mock()
    transformer2.transform.return_value = (context, state)

    result_ctx, result_state = _apply_transformers(context, state, [transformer1, transformer2])

    assert result_ctx == context
    assert result_state == state
    transformer1.transform.assert_called_once()
    transformer2.transform.assert_called_once()


def test_apply_transformers_with_should_apply():
    """Skip transformer when should_apply returns False."""
    context = Mock()
    state = Mock()
    transformer = Mock()
    transformer.should_apply.return_value = False

    result_ctx, result_state = _apply_transformers(context, state, [transformer])

    assert result_ctx == context
    assert result_state == state
    transformer.transform.assert_not_called()


def test_collect_response_patterns():
    """Collect patterns from transformers."""
    state = Mock()
    t1 = Mock()
    t1.on_response.return_value = [{"pattern": "p1"}]
    t2 = Mock()
    t2.on_response.return_value = [{"pattern": "p2"}]

    patterns = _collect_response_patterns([t1, t2], state)

    assert len(patterns) == 2
    assert {"pattern": "p2"} in patterns
    assert {"pattern": "p1"} in patterns


def test_collect_response_patterns_no_on_response():
    """Handle transformers without on_response method."""
    state = Mock()
    transformer = Mock(spec=[])
    del transformer.on_response

    patterns = _collect_response_patterns([transformer], state)

    assert patterns == []
