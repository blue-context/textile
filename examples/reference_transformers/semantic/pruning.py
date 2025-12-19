"""Reference Implementation: Semantic Pruning Transformer

PATTERN: Relevance-Based Filtering
===================================
This transformer demonstrates how to filter messages based on semantic
similarity to the current query, removing off-topic messages.

TEACHING POINTS:
- Embedding-based similarity computation
- Semantic filtering without temporal considerations
- Fail-safe handling (keep at least one message)
- Error handling for embedding computation
- When pure semantic filtering can cause problems (topic shifts)

USE THIS AS A TEMPLATE FOR:
- Topic-focused applications (Q&A, documentation chat)
- Knowledge retrieval systems
- Content moderation based on relevance
- Query-focused context windows

WHEN TO USE:
✅ Topic-focused conversations (staying on subject)
✅ Q&A systems where old off-topic messages should be removed
✅ When conversation should remain coherent around a theme
✅ Knowledge base chat where relevance > recency

WHEN NOT TO USE:
❌ Conversations with frequent topic changes (causes thrashing)
❌ When all history matters regardless of topic
❌ Very short conversations (not enough context)
❌ When embeddings are expensive or unavailable

COMMON PITFALLS:
⚠️ Topic Thrashing: User changes subject, transformer removes all history
⚠️ Over-Pruning: Threshold too high, removes too much context
⚠️ Embedding Costs: Every message requires embedding computation
⚠️ Startup Cold: No history = nothing to prune

CUSTOMIZATION IDEAS:
- Combine with temporal decay (see SemanticDecayTransformer)
- Add topic change detection to preserve context across shifts
- Implement similarity to conversation theme, not just last query
- Use multi-vector similarity (similarity to multiple recent queries)
"""

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
        """Initialize semantic pruning transformer.

        Args:
            similarity_threshold: Minimum similarity to keep (0.0-1.0)

        Raises:
            ValueError: If threshold not in valid range
        """
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(f"similarity_threshold must be in [0, 1], got {similarity_threshold}")
        self.threshold = similarity_threshold

    def transform(
        self,
        context: ContextWindow,
        state: TurnState,
    ) -> tuple[ContextWindow, TurnState]:
        """Remove messages below similarity threshold.

        IMPLEMENTATION PATTERN:
        1. Get query embedding from state
        2. Compute similarity for each message
        3. Filter below threshold (with error handling)
        4. Ensure at least one message remains (fail-safe)
        5. Remove filtered messages (immutable)

        Args:
            context: Context window to transform
            state: Turn state containing query embedding

        Returns:
            Tuple of (transformed context, unchanged state)
        """
        query_embedding = state.user_embedding
        to_remove: list[str] = []

        # STEP 1: Compute similarity for each message
        for msg in context.messages:
            # Skip system messages (instructions always relevant)
            if msg.role == "system" or msg.embedding is None:
                continue

            try:
                # PATTERN: Cosine similarity between query and message
                similarity = cosine_similarity(query_embedding, msg.embedding)
                if similarity < self.threshold:
                    to_remove.append(msg.id)
            except (ValueError, RuntimeError) as e:
                # PATTERN: Fail-safe - keep message on error
                logger.warning(f"Failed to compute similarity for message {msg.id}: {e}")
                # Keep message on error (conservative approach)

        # STEP 2: Ensure at least one non-system message remains
        # This prevents catastrophic pruning when query is off-topic
        non_system_messages = [m for m in context.messages if m.role != "system"]
        non_system_ids = {m.id for m in non_system_messages}

        # Check if we would remove all non-system messages
        if non_system_ids and non_system_ids.issubset(set(to_remove)):
            # Keep the most recent non-system message as anchor
            most_recent = max(non_system_messages, key=lambda m: m.turn_index)
            to_remove.remove(most_recent.id)
            logger.warning(
                "Semantic pruning would remove all non-system messages, "
                f"keeping most recent: turn={most_recent.turn_index}"
            )

        # STEP 3: Remove filtered messages (IMMUTABLE PATTERN)
        for msg_id in to_remove:
            context.remove_message(msg_id)

        return context, state

    def should_apply(self, context: ContextWindow, state: TurnState) -> bool:
        """Apply only if messages have embeddings.

        PATTERN: Conditional execution based on prerequisites
        Don't run if embeddings aren't available.

        Args:
            context: Context window
            state: Turn state

        Returns:
            True if any message has embeddings
        """
        return any(msg.embedding is not None for msg in context.messages)
