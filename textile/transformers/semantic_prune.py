"""Semantic pruning transformer for removing irrelevant messages."""

import logging

from textile.core.context_window import ContextWindow
from textile.core.turn_state import TurnState
from textile.transformers.base import ContextTransformer
from textile.utils.similarity import cosine_similarity

logger = logging.getLogger(__name__)


class SemanticPruningTransformer(ContextTransformer):
    """Remove messages with low semantic similarity to query.

    Pure semantic filtering without temporal considerations.
    Can cause thrashing when topics shift - consider SemanticDecayTransformer.
    """

    def __init__(self, similarity_threshold: float = 0.3) -> None:
        """Initialize semantic pruning transformer."""
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(f"similarity_threshold must be in [0, 1], got {similarity_threshold}")
        self.threshold = similarity_threshold

    def transform(
        self,
        context: ContextWindow,
        state: TurnState,
    ) -> tuple[ContextWindow, TurnState]:
        """Remove messages below similarity threshold."""
        query_embedding = state.user_embedding
        to_remove: list[str] = []

        for msg in context.messages:
            if msg.role == "system" or msg.embedding is None:
                continue

            try:
                similarity = cosine_similarity(query_embedding, msg.embedding)
                if similarity < self.threshold:
                    to_remove.append(msg.id)
            except (ValueError, RuntimeError) as e:
                logger.warning(f"Failed to compute similarity for message {msg.id}: {e}")
                # Keep message on error (fail-safe)

        # Ensure at least one non-system message remains
        non_system_messages = [m for m in context.messages if m.role != "system"]
        non_system_ids = {m.id for m in non_system_messages}

        # Check if we would remove all non-system messages
        if non_system_ids and non_system_ids.issubset(set(to_remove)):
            # Keep the most recent non-system message
            most_recent = max(non_system_messages, key=lambda m: m.turn_index)
            to_remove.remove(most_recent.id)
            logger.warning(
                "Semantic pruning would remove all non-system messages, "
                f"keeping most recent: turn={most_recent.turn_index}"
            )

        for msg_id in to_remove:
            context.remove_message(msg_id)

        return context, state

    def should_apply(self, context: ContextWindow, state: TurnState) -> bool:
        """Apply only if messages have embeddings."""
        return any(msg.embedding is not None for msg in context.messages)
