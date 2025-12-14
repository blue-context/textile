"""Immutable turn state for transformer pipeline."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TurnState:
    """Immutable turn state for transformer pipeline.

    Stateless turn state with fields needed by transformers.
    No storage concepts (conversation_id, persistence).

    Example:
        >>> state = TurnState(
        ...     user_message="What is the weather?",
        ...     turn_index=5,
        ...     tools=[{"type": "function", "function": {...}}]
        ... )
    """

    user_message: str
    turn_index: int = 0
    user_embedding: list[float] | None = None
    tools: list[dict] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
