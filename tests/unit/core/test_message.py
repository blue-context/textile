"""Concise Message model tests."""

import pytest

from textile.core.message import Message
from textile.core.metadata import MessageMetadata


class TestMessageCreation:
    """Message creation and initialization."""

    @pytest.mark.parametrize("role", ["user", "assistant", "system", "tool"])
    def test_from_dict_supports_all_roles(self, role: str) -> None:
        msg = Message.from_dict({"role": role, "content": "test"})
        assert msg.role == role
        assert msg.content == "test"

    @pytest.mark.parametrize("invalid_role", ["admin", "bot", "", "USER"])
    def test_invalid_role_raises_error(self, invalid_role: str) -> None:
        with pytest.raises(ValueError, match="Invalid role"):
            Message(role=invalid_role, content="test")

    def test_from_dict_basic_fields(self, sample_message_dict: dict) -> None:
        msg = Message.from_dict(sample_message_dict)
        assert msg.role == sample_message_dict["role"]
        assert msg.content == sample_message_dict["content"]

    def test_from_dict_with_tool_calls(self, sample_tool_calls: list[dict]) -> None:
        data = {"role": "assistant", "content": "", "tool_calls": sample_tool_calls}
        msg = Message.from_dict(data)
        assert msg.tool_calls == sample_tool_calls

    def test_from_dict_with_tool_call_id(self) -> None:
        data = {"role": "tool", "content": "result", "tool_call_id": "call_123"}
        msg = Message.from_dict(data)
        assert msg.tool_call_id == "call_123"

    def test_generates_unique_id(self) -> None:
        msg1 = Message(role="user", content="test")
        msg2 = Message(role="user", content="test")
        assert msg1.id != msg2.id
        assert msg1.id.startswith("msg_")

    def test_default_metadata_created(self) -> None:
        msg = Message(role="user", content="test")
        assert isinstance(msg.metadata, MessageMetadata)


class TestMessageSerialization:
    """Message to_dict serialization."""

    def test_to_dict_basic_fields(self, sample_message: Message) -> None:
        data = sample_message.to_dict()
        assert data["role"] == sample_message.role
        assert data["content"] == sample_message.content
        assert "id" not in data
        assert "metadata" not in data

    def test_to_dict_with_tool_calls(self, sample_tool_calls: list[dict]) -> None:
        msg = Message(role="assistant", content="", tool_calls=sample_tool_calls)
        data = msg.to_dict()
        assert data["tool_calls"] == sample_tool_calls

    def test_to_dict_with_tool_call_id(self) -> None:
        msg = Message(role="tool", content="result", tool_call_id="call_123")
        data = msg.to_dict()
        assert data["tool_call_id"] == "call_123"

    def test_to_dict_excludes_none_tool_fields(self) -> None:
        msg = Message(role="user", content="test")
        data = msg.to_dict()
        assert "tool_calls" not in data
        assert "tool_call_id" not in data


class TestMessageMetadataProperties:
    """Message metadata property shortcuts."""

    def test_turn_index_property_getter(self) -> None:
        msg = Message(role="user", content="test")
        msg.metadata.turn_index = 5
        assert msg.turn_index == 5

    def test_turn_index_property_setter(self) -> None:
        msg = Message(role="user", content="test")
        msg.turn_index = 3
        assert msg.metadata.turn_index == 3

    @pytest.mark.parametrize(
        "embedding",
        [
            [0.1, 0.2, 0.3],
            [1.0] * 1536,
            None,
        ],
    )
    def test_embedding_property_roundtrip(self, embedding: list[float] | None) -> None:
        msg = Message(role="user", content="test")
        msg.embedding = embedding
        assert msg.embedding == embedding
