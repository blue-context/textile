"""Semantic tool selection transformer for large tool catalogs."""

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
            cache_embeddings: Cache embeddings (recommended)
        """
        if max_tools <= 0:
            raise ValueError(f"max_tools must be positive, got {max_tools}")

        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be in [0, 1], got {similarity_threshold}"
            )

        self.max_tools = max_tools
        self.threshold = similarity_threshold
        self.cache_embeddings = cache_embeddings
        self._embedding_cache: dict[str, np.ndarray] = {}

    def transform(
        self,
        context: ContextWindow,
        state: TurnState,
    ) -> tuple[ContextWindow, TurnState]:
        """Select most relevant tools by semantic similarity.

        Args:
            context: Context window
            state: Turn state (contains tools)

        Returns:
            Tuple of (context, state with filtered tools)

        Raises:
            ValueError: If embedding model is not configured
        """
        from dataclasses import replace

        tools = state.tools

        if not tools:
            return context, state

        query_embedding = state.user_embedding
        tool_scores: list[tuple[dict[str, Any], float]] = []

        # Check embedding model configuration early
        from textile.config import get_config

        config = get_config()
        if config.embedding_model is None:
            raise ValueError(
                "SemanticToolSelectionTransformer requires an embedding model. "
                "Configure via: textile.configure(embedding_model=Embedding('text-embedding-3-small'))"
            )

        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                tool_name = func.get("name", "")
                tool_desc = func.get("description", "")
                tool_text = f"{tool_name}: {tool_desc}"

                if self.cache_embeddings and tool_name in self._embedding_cache:
                    tool_embedding = self._embedding_cache[tool_name]
                else:
                    tool_embedding = config.embedding_model.encode(tool_text)

                    if self.cache_embeddings:
                        self._embedding_cache[tool_name] = tool_embedding

                similarity = cosine_similarity(query_embedding, tool_embedding)

                if similarity >= self.threshold:
                    tool_scores.append((tool, similarity))

        tool_scores.sort(key=lambda x: x[1], reverse=True)
        selected_tools = [tool for tool, _ in tool_scores[: self.max_tools]]

        if context.messages:
            selected_tool_names = [
                t.get("function", {}).get("name", "") for t in selected_tools if t.get("type") == "function"
            ]
            context.messages[0].metadata._set_raw("selected_tools", selected_tool_names)
            context.messages[0].metadata._set_raw("tools_filtered", len(tools) - len(selected_tools))

        new_state = replace(state, tools=selected_tools)
        return context, new_state

    def should_apply(self, context: ContextWindow, state: TurnState) -> bool:
        """Apply only if tools exceed threshold.

        Args:
            context: Context window
            state: Turn state

        Returns:
            True if tool count > max_tools
        """
        if not state.tools:
            return False

        return len(state.tools) > self.max_tools
