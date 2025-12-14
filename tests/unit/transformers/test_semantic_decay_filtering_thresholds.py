"""Tests for SemanticDecayTransformer threshold-based filtering."""

import numpy as np

from textile.core.context_window import ContextWindow
from textile.core.turn_state import TurnState
from textile.transformers.semantic_decay import SemanticDecayTransformer
from conftest import create_message


class TestSemanticDecayFilteringThresholds:
    """Tests for threshold-based message filtering."""

    def test_filters_by_relevance_threshold(self):
        messages = [
            create_message("user", "High", turn_index=9),
            create_message("user", "Low", turn_index=0),
        ]
        for msg in messages:
            msg.metadata.prominence = 1.0

        context = ContextWindow(messages, max_tokens=4096)
        state = TurnState(user_message="Query", turn_index=10)

        transformer = SemanticDecayTransformer(
            half_life_turns=2,
            threshold=0.2,
            semantic_weight=0.0,
            temporal_weight=1.0,
        )
        context, _ = transformer.transform(context, state)

        assert len(context.messages) == 1
        assert context.messages[0].content == "High"

    def test_filters_by_semantic_threshold_with_embeddings(self):
        query_embedding = np.ones(384).tolist()

        similar_msg = create_message("user", "Similar", turn_index=5, embedding=query_embedding)
        similar_msg.metadata.prominence = 1.0

        different_msg = create_message("user", "Different", turn_index=5, embedding=(-np.ones(384)).tolist())
        different_msg.metadata.prominence = 1.0

        context = ContextWindow([similar_msg, different_msg], max_tokens=4096)
        state = TurnState(user_message="Query", turn_index=6, user_embedding=query_embedding)

        transformer = SemanticDecayTransformer(threshold=0.1, semantic_threshold=0.5)
        context, _ = transformer.transform(context, state)

        assert len(context.messages) == 1
        assert context.messages[0].content == "Similar"

    def test_records_pruned_count_in_state(self):
        messages = [create_message("user", f"Msg{i}", turn_index=i) for i in range(5)]
        for msg in messages:
            msg.metadata.prominence = 1.0

        context = ContextWindow(messages, max_tokens=4096)
        state = TurnState(user_message="Query", turn_index=20)

        transformer = SemanticDecayTransformer(threshold=0.9)
        _, result_state = transformer.transform(context, state)

        assert "semantic_decay_pruned" in result_state.metadata
        assert result_state.metadata["semantic_decay_pruned"] > 0
