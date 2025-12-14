"""Message for LLM APIs with transformer support."""

import uuid
from dataclasses import dataclass, field
from typing import Any

from textile.core.metadata import MessageMetadata


@dataclass
class Message:
    """Message for LLM APIs with transformer support.

    Stateless message with fields needed by transformers but no storage concepts.
    IDs are ephemeral (generated per-session), no conversation_id or persistence.

    Example:
        >>> msg = Message(
        ...     role="user",
        ...     content="What is the weather?",
        ...     metadata=MessageMetadata()
        ... )
        >>> msg.to_dict()
        {'role': 'user', 'content': 'What is the weather?'}
    """

    role: str
    content: str
    metadata: MessageMetadata = field(default_factory=MessageMetadata)
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:8]}")

    @property
    def turn_index(self) -> int:
        """Get turn index from metadata."""
        return self.metadata.turn_index

    @turn_index.setter
    def turn_index(self, value: int) -> None:
        """Set turn index in metadata."""
        self.metadata.turn_index = value

    @property
    def embedding(self) -> list[float] | None:
        """Get embedding from metadata."""
        return self.metadata.embedding

    @embedding.setter
    def embedding(self, value: list[float] | None) -> None:
        """Set embedding in metadata."""
        self.metadata.embedding = value

    def to_dict(self) -> dict[str, Any]:
        """Convert to LLM API format (OpenAI/LiteLLM)."""
        result = {"role": self.role, "content": self.content}
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """Create from API response dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            tool_calls=data.get("tool_calls"),
            tool_call_id=data.get("tool_call_id"),
        )

    def __post_init__(self) -> None:
        """Validate role."""
        valid_roles = {"system", "user", "assistant", "tool"}
        if self.role not in valid_roles:
            raise ValueError(f"Invalid role '{self.role}'. Must be one of {valid_roles}")
