"""Mutable message container with token budget."""

from dataclasses import dataclass
from typing import Any

from textile.core.message import Message


@dataclass
class ContextWindow:
    """Message container with token budget.

    Provides operations but does NOT:
    - Enforce ordering (maintains insertion order)
    - Apply filtering (transformers handle this)
    - Count tokens automatically (use textile.lite.tokens)
    - Transform on render (simple pass-through)

    Transformers mutate messages directly via context.messages.
    """

    messages: list[Message]
    max_tokens: int

    def add_message(self, message: Message, position: int | None = None) -> None:
        """Add message at position."""
        if position is None:
            self.messages.append(message)
        else:
            self.messages.insert(position, message)

    def remove_message(self, message_id: str) -> bool:
        """Remove message by ID."""
        original_length = len(self.messages)
        self.messages = [msg for msg in self.messages if msg.id != message_id]
        return len(self.messages) < original_length

    def get_message_by_id(self, message_id: str) -> Message | None:
        """Get message by ID."""
        for msg in self.messages:
            if msg.id == message_id:
                return msg
        return None

    def get_messages_by_role(self, role: str) -> list[Message]:
        """Get messages by role."""
        return [msg for msg in self.messages if msg.role == role]

    def render(self) -> list[dict]:
        """Convert to LLM API format (pass-through)."""
        return [msg.to_dict() for msg in self.messages]

    def total_tokens(
        self,
        model: str = "gpt-3.5-turbo",
        custom_tokenizer: Any | None = None
    ) -> int:
        """Count tokens using model-specific tokenizer."""
        from textile.lite.tokens import count_tokens

        message_dicts = [msg.to_dict() for msg in self.messages]
        return count_tokens(
            model=model,
            messages=message_dicts,
            custom_tokenizer=custom_tokenizer,
        )

    def __post_init__(self) -> None:
        """Validate max_tokens."""
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
