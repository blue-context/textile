"""Tests for SemanticDecayTransformer relevance calculation."""

from conftest import create_message

from textile.core.context_window import ContextWindow
from textile.core.turn_state import TurnState
from textile.transformers.semantic_decay import SemanticDecayMetadata, SemanticDecayTransformer


class TestSemanticDecayRelevance:
    """Tests for relevance calculation."""

    def test_calculates_temporal_decay(self):
        msg = create_message("user", "Old message", turn_index=0)
        msg.metadata.prominence = 1.0

        context = ContextWindow([msg], max_tokens=4096)
        state = TurnState(user_message="Query", turn_index=5)

        transformer = SemanticDecayTransformer(half_life_turns=5)
        context, _ = transformer.transform(context, state)

        assert msg.metadata.prominence < 1.0

    def test_applies_message_type_modifier(self):
        system_msg = create_message("system", "System", turn_index=5)
        system_msg.metadata.prominence = 1.0

        user_msg = create_message("user", "User", turn_index=5)
        user_msg.metadata.prominence = 1.0

        context = ContextWindow([system_msg, user_msg], max_tokens=4096)
        state = TurnState(user_message="Query", turn_index=6)

        transformer = SemanticDecayTransformer()
        context, _ = transformer.transform(context, state)

        assert system_msg.metadata.prominence > user_msg.metadata.prominence

    def test_applies_salience_boost(self):
        high_salience = create_message("user", "Important", turn_index=5)
        high_salience.metadata.prominence = 1.0
        high_meta = SemanticDecayMetadata(salience=1.0)
        high_salience.metadata.set_namespace("semantic_decay", high_meta)

        low_salience = create_message("user", "Unimportant", turn_index=5)
        low_salience.metadata.prominence = 1.0
        low_meta = SemanticDecayMetadata(salience=0.0)
        low_salience.metadata.set_namespace("semantic_decay", low_meta)

        context = ContextWindow([high_salience, low_salience], max_tokens=4096)
        state = TurnState(user_message="Query", turn_index=6)

        transformer = SemanticDecayTransformer()
        context, _ = transformer.transform(context, state)

        assert high_salience.metadata.prominence > low_salience.metadata.prominence

    def test_applies_recency_boost(self):
        recently_accessed = create_message("user", "Recent", turn_index=0)
        recently_accessed.metadata.prominence = 1.0
        recent_meta = SemanticDecayMetadata(last_access_turn=4)
        recently_accessed.metadata.set_namespace("semantic_decay", recent_meta)

        old_accessed = create_message("user", "Old", turn_index=0)
        old_accessed.metadata.prominence = 1.0
        old_meta = SemanticDecayMetadata(last_access_turn=0)
        old_accessed.metadata.set_namespace("semantic_decay", old_meta)

        context = ContextWindow([recently_accessed, old_accessed], max_tokens=4096)
        state = TurnState(user_message="Query", turn_index=5)

        transformer = SemanticDecayTransformer(recency_threshold=10)
        context, _ = transformer.transform(context, state)

        assert recently_accessed.metadata.prominence > old_accessed.metadata.prominence

    def test_stores_decay_components(self):
        msg = create_message("user", "Test", turn_index=0)
        msg.metadata.prominence = 1.0

        context = ContextWindow([msg], max_tokens=4096)
        state = TurnState(user_message="Query", turn_index=5)

        transformer = SemanticDecayTransformer()
        context, _ = transformer.transform(context, state)

        components = msg.metadata._get_raw("decay_components")
        assert components is not None
        assert "R0" in components
        assert "m_type" in components
        assert "D_temporal" in components
        assert "age_turns" in components
