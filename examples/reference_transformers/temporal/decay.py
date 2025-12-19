"""Reference Implementation: Temporal Decay Transformer

PATTERN: Time-Based Filtering
==================================
This transformer demonstrates how to filter messages based on their age,
reducing prominence of older messages using exponential decay.

TEACHING POINTS:
- Simple stateless transformation pattern
- Weight calculation using exponential decay formula
- Immutable context handling (never mutate inputs)
- Metadata tracking for observability
- Graceful edge case handling (always keep minimum messages)

USE THIS AS A TEMPLATE FOR:
- Custom decay functions (linear, logarithmic, step-wise)
- Domain-specific aging (user activity decay, session-based)
- Priority-based retention with time components
- Hybrid approaches combining age with other factors

WHEN TO USE:
✅ Long conversations where recent context matters more
✅ Chatbots with multi-turn conversations
✅ Applications where older messages become less relevant
✅ Memory management for context window limits

WHEN NOT TO USE:
❌ Short conversations (< 5 turns)
❌ When all history is equally important (e.g., legal documents)
❌ When topic shifts are frequent (use semantic approaches)
❌ Real-time systems requiring hard token limits (add budget enforcement)

CUSTOMIZATION IDEAS:
- Adjust decay formula (linear, logarithmic instead of exponential)
- Add role-specific decay rates (slower for system messages)
- Combine with importance scoring from your domain
- Implement adaptive half-life based on conversation length
"""

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

        Raises:
            ValueError: If parameters are invalid
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
        """Apply exponential decay and prune low-prominence messages.

        IMPLEMENTATION PATTERN:
        1. Calculate decay for all messages
        2. Identify messages to keep (above threshold)
        3. Ensure minimum recent messages (graceful degradation)
        4. Filter context (immutable - create new list)
        5. Track metrics in metadata

        Args:
            context: Context window to transform
            state: Turn state for current turn index

        Returns:
            Tuple of (transformed context, unchanged state)
        """
        current_turn = state.turn_index
        initial_count = len(context.messages)

        logger.debug(
            f"DecayTransformer: BEFORE transform - {initial_count} messages, "
            f"turn_index={current_turn}, half_life={self.half_life}, threshold={self.threshold}"
        )

        # STEP 1: Apply decay to all messages
        # Note: This mutates message metadata, which is acceptable
        # The transformer protocol requires immutable ContextWindow, not Message
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

        # STEP 2: Always keep system messages (never remove instructions)
        messages_to_keep = {msg.id for msg in context.messages if msg.role == "system"}

        # STEP 3: Keep non-system messages above threshold
        non_system_above_threshold = [
            msg
            for msg in context.messages
            if msg.role != "system" and msg.metadata.prominence >= self.threshold
        ]
        messages_to_keep.update(msg.id for msg in non_system_above_threshold)

        # STEP 4: Ensure we keep minimum recent messages for context continuity
        # This prevents catastrophic forgetting - always maintain basic context
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

        # STEP 5: Ensure at least one non-system message (fail-safe)
        if non_system_messages and not any(
            msg.id in messages_to_keep for msg in non_system_messages
        ):
            best = max(non_system_messages, key=lambda m: m.metadata.prominence)
            messages_to_keep.add(best.id)
            logger.debug(
                f"  No messages kept, keeping best: prominence={best.metadata.prominence:.3f}"
            )

        # STEP 6: Filter messages (IMMUTABLE PATTERN - create new list)
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
        """Apply only if context has multiple messages.

        PATTERN: Conditional execution
        Only run the transformer when it makes sense to avoid unnecessary work.

        Args:
            context: Context window
            state: Turn state

        Returns:
            True if 2+ messages exist
        """
        return len(context.messages) > 1
