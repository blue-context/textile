"""Decay transformer for reducing prominence of old messages."""

import logging

from textile.core.context_window import ContextWindow
from textile.core.turn_state import TurnState
from textile.transformers.base import ContextTransformer

logger = logging.getLogger(__name__)


class DecayTransformer(ContextTransformer):
    """Apply exponential decay and prune low-prominence messages.

    Formula: prominence * 0.5^(age / half_life)
    System messages never removed.
    Always keeps minimum recent messages for context continuity.
    """

    def __init__(
        self,
        half_life_turns: int = 5,
        threshold: float = 0.1,
        min_recent_messages: int = 10,
    ) -> None:
        """Initialize decay transformer.

        Args:
            half_life_turns: Number of turns for prominence to decay by half
            threshold: Minimum prominence to keep (0.0-1.0)
            min_recent_messages: Minimum number of recent messages to always keep
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be between 0.0 and 1.0, got {threshold}")
        if min_recent_messages < 1:
            raise ValueError(f"min_recent_messages must be >= 1, got {min_recent_messages}")

        self.half_life = half_life_turns
        self.threshold = threshold
        self.min_recent_messages = min_recent_messages

    def transform(
        self,
        context: ContextWindow,
        state: TurnState,
    ) -> tuple[ContextWindow, TurnState]:
        """Apply exponential decay and prune low-prominence messages."""
        current_turn = state.turn_index
        initial_count = len(context.messages)

        logger.debug(
            f"DecayTransformer: BEFORE transform - {initial_count} messages, "
            f"turn_index={current_turn}, half_life={self.half_life}, threshold={self.threshold}"
        )

        # Apply decay to all messages
        for msg in context.messages:
            age_turns = current_turn - msg.turn_index
            decay_factor = 0.5 ** (age_turns / self.half_life)
            old_prominence = msg.metadata.prominence
            msg.metadata.prominence *= decay_factor

            logger.debug(
                f"  Message turn={msg.turn_index}, age={age_turns}, "
                f"role={msg.role}, prominence: {old_prominence:.3f} -> {msg.metadata.prominence:.3f}, "
                f"content_preview={msg.content[:50]!r}..."
            )

        # Always keep system messages
        messages_to_keep = {msg.id for msg in context.messages if msg.role == "system"}

        # Keep non-system above threshold
        non_system_above_threshold = [
            msg for msg in context.messages
            if msg.role != "system" and msg.metadata.prominence >= self.threshold
        ]
        messages_to_keep.update(msg.id for msg in non_system_above_threshold)

        # Ensure we keep minimum recent messages for context continuity
        # Get all non-system messages sorted by turn index (most recent first)
        non_system_messages = [msg for msg in context.messages if msg.role != "system"]
        non_system_messages.sort(key=lambda m: m.turn_index, reverse=True)

        # Guarantee the last N messages are kept
        for i, msg in enumerate(non_system_messages):
            if i < self.min_recent_messages:
                if msg.id not in messages_to_keep:
                    messages_to_keep.add(msg.id)
                    logger.debug(
                        f"  Added recent message (min_recent guarantee): turn={msg.turn_index}, "
                        f"prominence={msg.metadata.prominence:.3f}"
                    )
            else:
                break

        # Ensure at least one non-system message (should already be satisfied)
        if non_system_messages and not any(msg.id in messages_to_keep for msg in non_system_messages):
            best = max(non_system_messages, key=lambda m: m.metadata.prominence)
            messages_to_keep.add(best.id)
            logger.debug(
                f"  No messages kept, keeping best: "
                f"prominence={best.metadata.prominence:.3f}"
            )

        # Filter messages
        filtered_messages = [msg for msg in context.messages if msg.id not in messages_to_keep]
        context.messages = [msg for msg in context.messages if msg.id in messages_to_keep]

        logger.debug(
            f"DecayTransformer: AFTER transform - {len(context.messages)} messages kept, "
            f"{len(filtered_messages)} filtered"
        )

        if filtered_messages:
            for msg in filtered_messages:
                logger.debug(
                    f"  FILTERED: turn={msg.turn_index}, role={msg.role}, "
                    f"prominence={msg.metadata.prominence:.3f}, "
                    f"content={msg.content[:50]!r}..."
                )

        return context, state

    def should_apply(self, context: ContextWindow, state: TurnState) -> bool:
        """Apply only if context has multiple messages."""
        return len(context.messages) > 1
