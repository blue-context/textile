"""Tests for SemanticPruningTransformer."""

import numpy as np
import pytest

from textile.core.context_window import ContextWindow
from textile.core.message import Message
from textile.core.turn_state import TurnState
from textile.transformers.semantic_prune import SemanticPruningTransformer
from conftest import create_message


class TestSemanticPruningTransformer:
    """Tests for semantic pruning transformer."""

    @pytest.mark.parametrize("threshold,valid", [
        (0.0, True), (0.5, True), (1.0, True),
        (-0.1, False), (1.1, False),
    ])
    def test_init_validates_threshold(self, threshold, valid):
        if valid:
            transformer = SemanticPruningTransformer(similarity_threshold=threshold)
            assert transformer.threshold == threshold
        else:
            with pytest.raises(ValueError, match="similarity_threshold must be in"):
                SemanticPruningTransformer(similarity_threshold=threshold)

    def test_removes_low_similarity_messages(self):
        query_embedding = np.ones(384).tolist()

        similar_msg = create_message("user", "Similar", embedding=query_embedding)
        different_msg = create_message("user", "Different", embedding=(-np.ones(384)).tolist())

        context = ContextWindow([similar_msg, different_msg], max_tokens=4096)
        state = TurnState(user_message="Query", user_embedding=query_embedding)
        transformer = SemanticPruningTransformer(similarity_threshold=0.8)

        context, _ = transformer.transform(context, state)

        assert len(context.messages) == 1
        assert context.messages[0].content == "Similar"

    def test_keeps_system_messages_always(self):
        query_embedding = np.ones(384).tolist()

        system_msg = create_message("system", "System")
        user_msg = create_message("user", "User", embedding=(-np.ones(384)).tolist())

        context = ContextWindow([system_msg, user_msg], max_tokens=4096)
        state = TurnState(user_message="Query", user_embedding=query_embedding)
        transformer = SemanticPruningTransformer(similarity_threshold=0.9)

        context, _ = transformer.transform(context, state)

        assert any(m.role == "system" for m in context.messages)

    def test_skips_messages_without_embeddings(self):
        query_embedding = np.ones(384).tolist()

        with_msg = create_message("user", "With", embedding=(-np.ones(384)).tolist())
        without_msg = create_message("user", "Without")

        context = ContextWindow([with_msg, without_msg], max_tokens=4096)
        state = TurnState(user_message="Query", user_embedding=query_embedding)
        transformer = SemanticPruningTransformer(similarity_threshold=0.9)

        context, _ = transformer.transform(context, state)

        without_embedding = [m for m in context.messages if m.content == "Without"]
        assert len(without_embedding) == 1

    def test_should_apply_requires_embeddings(self, sample_state):
        no_embeddings = ContextWindow([Message(role="user", content="Test")], max_tokens=4096)
        transformer = SemanticPruningTransformer()

        assert not transformer.should_apply(no_embeddings, sample_state)

    def test_should_apply_with_embeddings(self, sample_state):
        msg = create_message("user", "Test", embedding=np.random.rand(384).tolist())
        with_embeddings = ContextWindow([msg], max_tokens=4096)
        transformer = SemanticPruningTransformer()

        assert transformer.should_apply(with_embeddings, sample_state)
