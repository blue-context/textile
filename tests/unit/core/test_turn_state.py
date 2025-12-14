"""Concise TurnState tests."""

import pytest

from textile.core.turn_state import TurnState


class TestTurnStateCreation:
    """TurnState creation and immutability."""

    def test_basic_creation(self) -> None:
        state = TurnState(user_message="Hello")
        assert state.user_message == "Hello"
        assert state.turn_index == 0
        assert state.user_embedding is None
        assert state.tools is None
        assert state.metadata == {}

    @pytest.mark.parametrize("turn_index", [0, 1, 5, 100])
    def test_with_turn_index(self, turn_index: int) -> None:
        state = TurnState(user_message="test", turn_index=turn_index)
        assert state.turn_index == turn_index

    def test_with_embedding(self) -> None:
        embedding = [0.1, 0.2, 0.3]
        state = TurnState(user_message="test", user_embedding=embedding)
        assert state.user_embedding == embedding

    def test_with_tools(self) -> None:
        tools = [{"type": "function", "function": {"name": "test"}}]
        state = TurnState(user_message="test", tools=tools)
        assert state.tools == tools

    def test_with_metadata(self) -> None:
        metadata = {"key": "value", "nested": {"data": 123}}
        state = TurnState(user_message="test", metadata=metadata)
        assert state.metadata == metadata

    def test_immutability(self) -> None:
        state = TurnState(user_message="test")
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            state.user_message = "changed"  # type: ignore[misc]


class TestTurnStateDefaults:
    """Default value behavior."""

    def test_default_turn_index_is_zero(self) -> None:
        state = TurnState(user_message="test")
        assert state.turn_index == 0

    def test_default_embedding_is_none(self) -> None:
        state = TurnState(user_message="test")
        assert state.user_embedding is None

    def test_default_tools_is_none(self) -> None:
        state = TurnState(user_message="test")
        assert state.tools is None

    def test_default_metadata_is_empty_dict(self) -> None:
        state = TurnState(user_message="test")
        assert state.metadata == {}
        assert isinstance(state.metadata, dict)


class TestTurnStateFixture:
    """Test using sample fixture."""

    def test_sample_fixture_structure(self, sample_turn_state: TurnState) -> None:
        assert sample_turn_state.user_message == "Test message"
        assert sample_turn_state.turn_index == 0
        assert sample_turn_state.user_embedding == [0.1, 0.2, 0.3]
        assert sample_turn_state.tools is None
        assert sample_turn_state.metadata == {}
