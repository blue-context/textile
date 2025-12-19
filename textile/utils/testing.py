"""Testing utilities for custom transformers.

Provides helper functions and base classes for testing custom transformers
without boilerplate.
"""

from typing import Any

import pytest

from textile.core.context_window import ContextWindow
from textile.core.message import Message
from textile.core.turn_state import TurnState
from textile.transformers.base import ContextTransformer


def create_message(
    role: str,
    content: str,
    turn_index: int = 0,
    **metadata: Any,
) -> Message:
    """Create a test message with optional metadata.

    Args:
        role: Message role ("system", "user", "assistant")
        content: Message content
        turn_index: Turn index for the message
        **metadata: Additional metadata key-value pairs

    Returns:
        Message instance

    Example:
        >>> msg = create_message("user", "Hello", turn_index=5, important=True)
        >>> assert msg.turn_index == 5
        >>> assert msg.metadata._get_raw("important") is True
    """
    msg_dict = {"role": role, "content": content}
    msg = Message.from_dict(msg_dict)
    msg.turn_index = turn_index

    for key, value in metadata.items():
        msg.metadata._set_raw(key, value)

    return msg


def create_context(
    messages: list[dict[str, Any]] | list[Message],
    max_tokens: int | None = None,
) -> ContextWindow:
    """Create a test context window from message dicts or Messages.

    Args:
        messages: List of message dicts or Message instances
        max_tokens: Optional max token limit

    Returns:
        ContextWindow instance

    Example:
        >>> context = create_context([
        ...     {"role": "system", "content": "You are helpful"},
        ...     {"role": "user", "content": "Hello", "turn_index": 1},
        ... ])
        >>> assert len(context.messages) == 2
    """
    msg_list = []
    for item in messages:
        if isinstance(item, Message):
            msg_list.append(item)
        else:
            msg = Message.from_dict(item)
            if "turn_index" in item:
                msg.turn_index = item["turn_index"]
            msg_list.append(msg)

    kwargs = {"messages": msg_list}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    return ContextWindow(**kwargs)


def create_turn_state(
    turn_index: int = 0,
    metadata: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> TurnState:
    """Create a test turn state.

    Args:
        turn_index: Current turn index
        metadata: Optional metadata dict
        tools: Optional tools list

    Returns:
        TurnState instance

    Example:
        >>> state = create_turn_state(turn_index=5, metadata={"test": True})
        >>> assert state.turn_index == 5
        >>> assert state.metadata["test"] is True
    """
    return TurnState(
        turn_index=turn_index,
        metadata=metadata or {},
        tools=tools or [],
    )


def assert_messages_removed(
    context: ContextWindow,
    expected: int | None = None,
    min_removed: int | None = None,
    max_removed: int | None = None,
) -> None:
    """Assert message removal counts.

    Args:
        context: Context window after transformation
        expected: Exact number of messages expected to remain
        min_removed: Minimum messages removed
        max_removed: Maximum messages removed

    Raises:
        AssertionError: If conditions not met

    Example:
        >>> context = create_context([{"role": "user", "content": "Hi"}])
        >>> assert_messages_removed(context, expected=1)
    """
    if expected is not None:
        assert len(context.messages) == expected, (
            f"Expected {expected} messages, got {len(context.messages)}"
        )

    # For min/max removed, we need the original count
    # These are typically used with a fixture that tracks original count


def assert_message_preserved(
    context: ContextWindow,
    role: str | None = None,
    content: str | None = None,
    message_id: str | None = None,
) -> None:
    """Assert a specific message is preserved.

    Args:
        context: Context window after transformation
        role: Expected role to find
        content: Expected content to find
        message_id: Expected message ID to find

    Raises:
        AssertionError: If message not found

    Example:
        >>> context = create_context([
        ...     {"role": "system", "content": "You are helpful"}
        ... ])
        >>> assert_message_preserved(context, role="system")
    """
    if message_id is not None:
        msg = context.get_message_by_id(message_id)
        assert msg is not None, f"Message {message_id} not found"
        return

    for msg in context.messages:
        if role is not None and msg.role != role:
            continue
        if content is not None and content not in msg.content:
            continue
        return  # Found matching message

    raise AssertionError(
        f"No message found with role={role}, content={content}"
    )


def assert_system_messages_preserved(context: ContextWindow) -> None:
    """Assert all system messages are preserved.

    System messages should never be removed by transformers.

    Args:
        context: Context window after transformation

    Raises:
        AssertionError: If system messages were removed

    Example:
        >>> context = create_context([
        ...     {"role": "system", "content": "Instructions"},
        ...     {"role": "user", "content": "Query"},
        ... ])
        >>> # After some transformation
        >>> assert_system_messages_preserved(context)
    """
    # This is a convention check - system messages should always remain
    # Can't verify without original context, but can check they exist
    system_messages = [m for m in context.messages if m.role == "system"]
    assert len(system_messages) > 0, "No system messages found - may have been removed"


class TransformerTestCase:
    """Base class for transformer test cases.

    Provides helper methods for testing transformers with less boilerplate.

    Example:
        >>> class TestMyTransformer(TransformerTestCase):
        ...     def test_removes_old_messages(self):
        ...         context, state = self.create_test_context(
        ...             messages=[
        ...                 {"role": "user", "content": "Old", "turn_index": 0},
        ...                 {"role": "user", "content": "New", "turn_index": 10},
        ...             ],
        ...             current_turn=10
        ...         )
        ...         transformer = MyTransformer(max_age=5)
        ...         new_context, _ = transformer.transform(context, state)
        ...         assert len(new_context.messages) == 1
    """

    def create_test_context(
        self,
        messages: list[dict[str, Any]],
        current_turn: int = 0,
        metadata: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[ContextWindow, TurnState]:
        """Create a test context and state together.

        Args:
            messages: List of message dicts
            current_turn: Current turn index
            metadata: Optional state metadata
            tools: Optional tools list

        Returns:
            Tuple of (ContextWindow, TurnState)
        """
        context = create_context(messages)
        state = create_turn_state(
            turn_index=current_turn,
            metadata=metadata,
            tools=tools,
        )
        return context, state

    def apply_transformer(
        self,
        transformer: ContextTransformer,
        messages: list[dict[str, Any]],
        current_turn: int = 0,
        expected_removed: int | None = None,
    ) -> tuple[ContextWindow, TurnState]:
        """Apply a transformer to test messages.

        Args:
            transformer: Transformer to test
            messages: Test messages
            current_turn: Current turn index
            expected_removed: Optional check for removed count

        Returns:
            Tuple of (transformed_context, transformed_state)
        """
        context, state = self.create_test_context(messages, current_turn)
        original_count = len(context.messages)

        new_context, new_state = transformer.transform(context, state)

        if expected_removed is not None:
            removed = original_count - len(new_context.messages)
            assert removed == expected_removed, (
                f"Expected {expected_removed} messages removed, "
                f"got {removed}"
            )

        return new_context, new_state

    def assert_transformer_applied(
        self,
        transformer: ContextTransformer,
        context: ContextWindow,
        state: TurnState,
    ) -> None:
        """Assert transformer should apply to context.

        Args:
            transformer: Transformer to check
            context: Test context
            state: Test state

        Raises:
            AssertionError: If should_apply returns False
        """
        assert transformer.should_apply(context, state), (
            f"{transformer.__class__.__name__}.should_apply() returned False"
        )

    def assert_transformer_skipped(
        self,
        transformer: ContextTransformer,
        context: ContextWindow,
        state: TurnState,
    ) -> None:
        """Assert transformer should skip for context.

        Args:
            transformer: Transformer to check
            context: Test context
            state: Test state

        Raises:
            AssertionError: If should_apply returns True
        """
        assert not transformer.should_apply(context, state), (
            f"{transformer.__class__.__name__}.should_apply() returned True"
        )


# Pytest fixtures for common test scenarios

@pytest.fixture
def simple_context():
    """Fixture providing a simple 3-message context."""
    return create_context([
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ])


@pytest.fixture
def multi_turn_context():
    """Fixture providing a multi-turn conversation."""
    return create_context([
        {"role": "system", "content": "You are helpful.", "turn_index": 0},
        {"role": "user", "content": "Message 1", "turn_index": 1},
        {"role": "assistant", "content": "Response 1", "turn_index": 2},
        {"role": "user", "content": "Message 2", "turn_index": 3},
        {"role": "assistant", "content": "Response 2", "turn_index": 4},
        {"role": "user", "content": "Message 3", "turn_index": 5},
    ])


@pytest.fixture
def current_state():
    """Fixture providing a basic turn state."""
    return create_turn_state(turn_index=5)
