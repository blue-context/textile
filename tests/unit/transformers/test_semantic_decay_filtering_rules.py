"""Tests for SemanticDecayTransformer filtering rules."""

from textile.core.context_window import ContextWindow
from textile.core.message import Message
from textile.core.turn_state import TurnState
from textile.transformers.semantic_decay import SemanticDecayTransformer
from conftest import create_message


class TestSemanticDecayFilteringRules:
    """Tests for system/non-system message filtering rules."""

    def test_always_keeps_system_messages(self):
        system_msg = create_message("system", "System", turn_index=0)
        system_msg.metadata.prominence = 0.01
        user_msg = create_message("user", "User", turn_index=9)

        context = ContextWindow([system_msg, user_msg], max_tokens=4096)
        state = TurnState(user_message="Query", turn_index=10)

        transformer = SemanticDecayTransformer(threshold=0.9)
        context, _ = transformer.transform(context, state)

        assert any(m.role == "system" for m in context.messages)

    def test_keeps_at_least_one_non_system_message(self):
        messages = [
            create_message("system", "System", turn_index=0),
            create_message("user", "Old1", turn_index=0),
            create_message("user", "Old2", turn_index=0),
        ]
        for msg in messages:
            msg.metadata.prominence = 1.0

        context = ContextWindow(messages, max_tokens=4096)
        state = TurnState(user_message="Query", turn_index=20)

        transformer = SemanticDecayTransformer(threshold=0.99)
        context, _ = transformer.transform(context, state)

        non_system = [m for m in context.messages if m.role != "system"]
        assert len(non_system) >= 1

    def test_should_apply_requires_multiple_messages(self):
        transformer = SemanticDecayTransformer()

        single = ContextWindow([Message(role="user", content="One")], max_tokens=4096)
        state = TurnState(user_message="Query", turn_index=1)
        assert not transformer.should_apply(single, state)

        multiple = ContextWindow([
            Message(role="user", content="One"),
            Message(role="user", content="Two"),
        ], max_tokens=4096)
        assert transformer.should_apply(multiple, state)

    def test_handles_empty_context(self):
        context = ContextWindow([], max_tokens=4096)
        state = TurnState(user_message="Query", turn_index=1)

        transformer = SemanticDecayTransformer()
        result_context, result_state = transformer.transform(context, state)

        assert len(result_context.messages) == 0
        assert result_state == state
