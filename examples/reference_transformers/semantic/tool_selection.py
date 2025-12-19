"""Reference Implementation: Semantic Tool Selection Transformer

PATTERN: State Transformation (Not Message Filtering)
======================================================
This transformer demonstrates how to modify TurnState rather than messages,
filtering tools based on semantic relevance to the query.

TEACHING POINTS:
- Transformer can modify state, not just context
- Caching pattern for expensive operations (embeddings)
- Working with tool definitions from LLM APIs
- Returning modified state (immutable pattern with dataclasses.replace)
- Conditional application based on tool count

USE THIS AS A TEMPLATE FOR:
- Tool/function filtering for large catalogs
- Dynamic capability selection
- Resource limiting (API quotas, token budgets)
- Context-aware feature toggling

WHEN TO USE:
✅ Large tool catalogs (50+ tools)
✅ Function calling with many available functions
✅ Token budget constraints with tool descriptions
✅ Improving LLM focus by reducing options

WHEN NOT TO USE:
❌ Small tool sets (< 10 tools)
❌ When all tools must always be available
❌ High-latency embeddings without caching
❌ Tools without good descriptions

PERFORMANCE NOTES:
- First call: Expensive (embed all tools)
- Subsequent calls: Fast (cached embeddings)
- Trade-off: Memory (cache) vs CPU (re-embedding)

CUSTOMIZATION IDEAS:
- Add tool usage history (boost frequently used tools)
- Multi-query similarity (relevant to conversation, not just last query)
- Category-based pre-filtering before semantic scoring
- Adaptive max_tools based on token budget
"""

from typing import Any

import numpy as np

from textile.core.context_window import ContextWindow
from textile.core.turn_state import TurnState
from textile.transformers.base import ContextTransformer
from textile.utils.similarity import cosine_similarity


class SemanticToolSelectionTransformer(ContextTransformer):
    """Filter tools to most relevant using semantic similarity.

    Useful for large tool catalogs (50+) to reduce tokens and improve focus.

    Embeds tool descriptions, computes similarity to query, returns top-k tools.
    """

    def __init__(
        self,
        max_tools: int = 10,
        similarity_threshold: float = 0.2,
        cache_embeddings: bool = True,
    ) -> None:
        """Initialize tool selection transformer.

        Args:
            max_tools: Maximum tools to select
            similarity_threshold: Minimum similarity to include
            cache_embeddings: Cache tool embeddings (recommended for production)

        Raises:
            ValueError: If parameters invalid
        """
        if max_tools <= 0:
            raise ValueError(f"max_tools must be positive, got {max_tools}")

        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(f"similarity_threshold must be in [0, 1], got {similarity_threshold}")

        self.max_tools = max_tools
        self.threshold = similarity_threshold
        self.cache_embeddings = cache_embeddings

        # PATTERN: Instance-level cache for expensive operations
        self._embedding_cache: dict[str, np.ndarray] = {}

    def transform(
        self,
        context: ContextWindow,
        state: TurnState,
    ) -> tuple[ContextWindow, TurnState]:
        """Select most relevant tools by semantic similarity.

        IMPLEMENTATION PATTERN:
        1. Get tools from state (not context!)
        2. Embed tool descriptions (with caching)
        3. Compute similarity to query
        4. Filter by threshold and top-k
        5. Return new state with filtered tools

        Args:
            context: Context window (unchanged)
            state: Turn state containing tools

        Returns:
            Tuple of (unchanged context, state with filtered tools)

        Raises:
            ValueError: If embedding model is not configured
        """
        from dataclasses import replace

        tools = state.tools

        if not tools:
            return context, state

        query_embedding = state.user_embedding
        tool_scores: list[tuple[dict[str, Any], float]] = []

        # PATTERN: Check configuration early, fail fast
        from textile.config import get_config

        config = get_config()
        if config.embedding_model is None:
            raise ValueError(
                "SemanticToolSelectionTransformer requires an embedding model. "
                "Configure via: textile.configure(embedding_model=Embedding('text-embedding-3-small'))"
            )

        # STEP 1: Score each tool
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                tool_name = func.get("name", "")
                tool_desc = func.get("description", "")
                tool_text = f"{tool_name}: {tool_desc}"

                # PATTERN: Caching expensive operations
                if self.cache_embeddings and tool_name in self._embedding_cache:
                    tool_embedding = self._embedding_cache[tool_name]
                else:
                    tool_embedding = config.embedding_model.encode(tool_text)

                    if self.cache_embeddings:
                        self._embedding_cache[tool_name] = tool_embedding

                similarity = cosine_similarity(query_embedding, tool_embedding)

                # Only consider tools above threshold
                if similarity >= self.threshold:
                    tool_scores.append((tool, similarity))

        # STEP 2: Sort by similarity and take top-k
        tool_scores.sort(key=lambda x: x[1], reverse=True)
        selected_tools = [tool for tool, _ in tool_scores[: self.max_tools]]

        # STEP 3: Track metrics in message metadata (for observability)
        if context.messages:
            selected_tool_names = [
                t.get("function", {}).get("name", "")
                for t in selected_tools
                if t.get("type") == "function"
            ]
            context.messages[0].metadata._set_raw("selected_tools", selected_tool_names)
            context.messages[0].metadata._set_raw(
                "tools_filtered", len(tools) - len(selected_tools)
            )

        # STEP 4: Return new state (IMMUTABLE PATTERN with dataclasses.replace)
        new_state = replace(state, tools=selected_tools)
        return context, new_state

    def should_apply(self, context: ContextWindow, state: TurnState) -> bool:
        """Apply only if tools exceed threshold.

        PATTERN: Conditional execution to avoid unnecessary work
        Only filter if tool count exceeds our limit.

        Args:
            context: Context window
            state: Turn state

        Returns:
            True if tool count > max_tools
        """
        if not state.tools:
            return False

        return len(state.tools) > self.max_tools
