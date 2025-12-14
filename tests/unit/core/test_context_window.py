"""Concise ContextWindow tests."""

import pytest

from textile.core.context_window import ContextWindow
from textile.core.message import Message


class TestContextWindowCreation:
    def test_creation_with_messages(self) -> None:
        messages = [Message(role="user", content="test")]
        cw = ContextWindow(messages=messages, max_tokens=100)
        assert len(cw.messages) == 1 and cw.max_tokens == 100

    def test_empty_context_window(self, empty_context_window: ContextWindow) -> None:
        assert len(empty_context_window.messages) == 0

    @pytest.mark.parametrize("max_tokens", [0, -1, -100])
    def test_invalid_max_tokens_raises_error(self, max_tokens: int) -> None:
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            ContextWindow(messages=[], max_tokens=max_tokens)

    @pytest.mark.parametrize("max_tokens", [1, 100, 4096, 128000])
    def test_valid_max_tokens(self, max_tokens: int) -> None:
        assert ContextWindow(messages=[], max_tokens=max_tokens).max_tokens == max_tokens


class TestAddMessage:
    def test_add_message_appends_by_default(self, empty_context_window: ContextWindow) -> None:
        msg = Message(role="user", content="test")
        empty_context_window.add_message(msg)
        assert empty_context_window.messages[0] == msg

    def test_add_message_at_position(self, sample_context_window: ContextWindow) -> None:
        msg = Message(role="system", content="inserted")
        sample_context_window.add_message(msg, position=0)
        assert sample_context_window.messages[0] == msg

    @pytest.mark.parametrize("position", [0, 1, 2])
    def test_add_at_various_positions(
        self, sample_context_window: ContextWindow, position: int
    ) -> None:
        msg = Message(role="system", content="new")
        sample_context_window.add_message(msg, position=position)
        assert sample_context_window.messages[position] == msg


class TestRemoveMessage:
    def test_remove_existing_message(self, sample_context_window: ContextWindow) -> None:
        msg_id = sample_context_window.messages[0].id
        assert sample_context_window.remove_message(msg_id) is True

    def test_remove_nonexistent_message(self, sample_context_window: ContextWindow) -> None:
        assert sample_context_window.remove_message("nonexistent") is False


class TestGetMessage:
    def test_get_message_by_id(self, sample_context_window: ContextWindow) -> None:
        msg = sample_context_window.messages[0]
        assert sample_context_window.get_message_by_id(msg.id) == msg

    def test_get_nonexistent_message(self, sample_context_window: ContextWindow) -> None:
        assert sample_context_window.get_message_by_id("nonexistent") is None

    @pytest.mark.parametrize("role,expected_count", [("user", 1), ("assistant", 1), ("system", 0)])
    def test_get_messages_by_role(
        self, sample_context_window: ContextWindow, role: str, expected_count: int
    ) -> None:
        messages = sample_context_window.get_messages_by_role(role)
        assert len(messages) == expected_count and all(msg.role == role for msg in messages)


class TestRender:
    def test_render_returns_dict_list(self, sample_context_window: ContextWindow) -> None:
        rendered = sample_context_window.render()
        assert all(isinstance(item, dict) for item in rendered)

    def test_render_preserves_order(self, sample_context_window: ContextWindow) -> None:
        rendered = sample_context_window.render()
        assert rendered[0]["content"] == sample_context_window.messages[0].content

    def test_render_empty_window(self, empty_context_window: ContextWindow) -> None:
        assert empty_context_window.render() == []


class TestTokenCounting:
    def test_total_tokens_basic(self, sample_context_window: ContextWindow) -> None:
        assert sample_context_window.total_tokens() > 0

    def test_total_tokens_empty(self, empty_context_window: ContextWindow) -> None:
        assert empty_context_window.total_tokens() >= 0
