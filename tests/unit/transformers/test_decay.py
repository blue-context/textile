"""Tests for DecayTransformer."""

import pytest

from textile.core.context_window import ContextWindow
from textile.core.message import Message
from textile.core.turn_state import TurnState
from textile.transformers.decay import DecayTransformer


class TestDecayTransformer:
    """Tests for exponential decay transformer."""

    def test_init_validates_threshold(self):
        """Test threshold validation."""
        assert DecayTransformer(threshold=0.0).threshold == 0.0
        assert DecayTransformer(threshold=0.5).threshold == 0.5
        assert DecayTransformer(threshold=1.0).threshold == 1.0

        with pytest.raises(ValueError, match="threshold must be between"):
            DecayTransformer(threshold=-0.1)
        with pytest.raises(ValueError, match="threshold must be between"):
            DecayTransformer(threshold=1.1)

    def test_init_validates_min_recent_messages(self):
        """Test min_recent_messages validation."""
        assert DecayTransformer(min_recent_messages=1).min_recent_messages == 1
        assert DecayTransformer(min_recent_messages=10).min_recent_messages == 10

        with pytest.raises(ValueError, match="min_recent_messages must be >= 1"):
            DecayTransformer(min_recent_messages=0)
        with pytest.raises(ValueError, match="min_recent_messages must be >= 1"):
            DecayTransformer(min_recent_messages=-1)

    @pytest.mark.parametrize(
        "half_life,msg_turn,state_turn,expected",
        [
            (5, 10, 10, 1.0),
            (5, 5, 10, 0.5),
            (5, 0, 10, 0.25),
            (10, 5, 15, 0.5),
        ],
    )
    def test_applies_exponential_decay(self, half_life, msg_turn, state_turn, expected):
        """Test that prominence decays exponentially based on age."""
        msg = Message(role="user", content="Test")
        msg.turn_index = msg_turn
        msg.metadata.prominence = 1.0
        context = ContextWindow([msg], max_tokens=4096)
        state = TurnState(user_message="Test", turn_index=state_turn)
        transformer = DecayTransformer(
            half_life_turns=half_life,
            threshold=0.0,
            min_recent_messages=1,  # Keep at least the message being tested
        )
        context, _ = transformer.transform(context, state)
        assert msg.metadata.prominence == pytest.approx(expected)

    def test_prunes_below_threshold(self):
        """Test that messages below threshold are pruned."""
        old_msg = Message(role="user", content="Old")
        old_msg.turn_index = 0
        recent_msg = Message(role="user", content="Recent")
        recent_msg.turn_index = 9
        for msg in [old_msg, recent_msg]:
            msg.metadata.prominence = 1.0
        context = ContextWindow([old_msg, recent_msg], max_tokens=4096)
        state = TurnState(user_message="Test", turn_index=10)
        transformer = DecayTransformer(
            half_life_turns=1,
            threshold=0.3,
            min_recent_messages=1,  # Only keep 1 message minimum
        )
        context, _ = transformer.transform(context, state)
        assert len(context.messages) == 1
        assert context.messages[0].content == "Recent"

    def test_always_keeps_system_messages(self):
        system_msg = Message(role="system", content="System")
        system_msg.turn_index = 0
        system_msg.metadata.prominence = 0.01
        user_msg = Message(role="user", content="User")
        user_msg.turn_index = 5
        user_msg.metadata.prominence = 1.0
        context = ContextWindow([system_msg, user_msg], max_tokens=4096)
        state = TurnState(user_message="Test", turn_index=10)
        transformer = DecayTransformer(threshold=0.5)
        context, _ = transformer.transform(context, state)
        assert len(context.messages) == 2
        assert any(msg.role == "system" for msg in context.messages)

    def test_keeps_at_least_one_non_system_message(self):
        """Test that at least one non-system message is always kept."""
        system_msg = Message(role="system", content="System")
        system_msg.turn_index = 0
        system_msg.metadata.prominence = 1.0
        old1_msg = Message(role="user", content="Old1")
        old1_msg.turn_index = 0
        old1_msg.metadata.prominence = 1.0
        old2_msg = Message(role="user", content="Old2")
        old2_msg.turn_index = 1
        old2_msg.metadata.prominence = 1.0
        context = ContextWindow([system_msg, old1_msg, old2_msg], max_tokens=4096)
        state = TurnState(user_message="Test", turn_index=20)
        transformer = DecayTransformer(
            half_life_turns=1,
            threshold=0.99,
            min_recent_messages=1,  # Minimum 1 message
        )
        context, _ = transformer.transform(context, state)
        non_system = [m for m in context.messages if m.role != "system"]
        assert len(non_system) >= 1

    def test_min_recent_messages_guarantee(self):
        """Test that min_recent_messages are always kept."""
        # Create 15 messages
        messages = []
        for i in range(15):
            msg = Message(role="user" if i % 2 == 0 else "assistant", content=f"Msg{i}")
            msg.turn_index = i
            msg.metadata.prominence = 1.0
            messages.append(msg)

        context = ContextWindow(messages, max_tokens=4096)
        state = TurnState(user_message="Test", turn_index=14)

        # Use aggressive threshold that would filter most messages
        transformer = DecayTransformer(
            half_life_turns=1,
            threshold=0.8,
            min_recent_messages=5,  # Guarantee last 5
        )
        context, _ = transformer.transform(context, state)

        # Check that we kept at least 5 messages
        assert len(context.messages) >= 5

        # Check that the most recent 5 are included
        kept_turns = {msg.turn_index for msg in context.messages}
        expected_recent = {10, 11, 12, 13, 14}  # Last 5 turns
        assert expected_recent.issubset(kept_turns)

    def test_should_apply_requires_multiple_messages(self, sample_state):
        transformer = DecayTransformer()
        single = ContextWindow([Message(role="user", content="One")], max_tokens=4096)
        assert not transformer.should_apply(single, sample_state)
        multiple = ContextWindow(
            [Message(role="user", content="One"), Message(role="user", content="Two")],
            max_tokens=4096,
        )
        assert transformer.should_apply(multiple, sample_state)
