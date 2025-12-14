"""Tests for SemanticDecayTransformer message type inference."""

from conftest import create_message

from textile.core.context_window import ContextWindow
from textile.core.message import Message
from textile.core.turn_state import TurnState
from textile.transformers.semantic_decay import MessageType, SemanticDecayTransformer


class TestSemanticDecayInference:
    """Tests for message type inference."""

    def test_infers_system_message_type(self):
        transformer = SemanticDecayTransformer()
        msg = Message(role="system", content="System prompt")

        msg_type = transformer._infer_message_type(msg)
        assert msg_type == MessageType.SYSTEM

    def test_infers_instruction_from_metadata(self):
        transformer = SemanticDecayTransformer()
        msg = Message(role="user", content="Do this")
        msg.metadata._set_raw("is_instruction", True)

        msg_type = transformer._infer_message_type(msg)
        assert msg_type == MessageType.INSTRUCTION

    def test_infers_factual_from_metadata(self):
        transformer = SemanticDecayTransformer()
        msg = Message(role="user", content="Fact")
        msg.metadata._set_raw("is_factual", True)

        msg_type = transformer._infer_message_type(msg)
        assert msg_type == MessageType.FACTUAL

    def test_infers_historical_from_metadata(self):
        transformer = SemanticDecayTransformer()
        msg = Message(role="user", content="Old")
        msg.metadata._set_raw("is_historical", True)

        msg_type = transformer._infer_message_type(msg)
        assert msg_type == MessageType.HISTORICAL

    def test_defaults_to_conversational(self):
        transformer = SemanticDecayTransformer()
        msg = Message(role="user", content="Regular message")

        msg_type = transformer._infer_message_type(msg)
        assert msg_type == MessageType.CONVERSATIONAL

    def test_creates_metadata_namespace_on_transform(self):
        msg = create_message("user", "Test", turn_index=0)
        msg.metadata.prominence = 1.0

        context = ContextWindow([msg], max_tokens=4096)
        state = TurnState(user_message="Query", turn_index=1)

        transformer = SemanticDecayTransformer()
        transformer.transform(context, state)

        assert msg.metadata.has_namespace("semantic_decay")
