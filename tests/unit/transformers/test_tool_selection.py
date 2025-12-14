"""Tests for SemanticToolSelectionTransformer."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from textile.core.turn_state import TurnState
from textile.transformers.tool_selection import SemanticToolSelectionTransformer


class TestSemanticToolSelectionTransformer:
    """Tests for semantic tool selection transformer."""

    @pytest.mark.parametrize("max_tools,valid", [
        (1, True), (10, True), (100, True), (0, False), (-1, False),
    ])
    def test_init_validates_max_tools(self, max_tools, valid):
        if valid:
            assert SemanticToolSelectionTransformer(max_tools=max_tools).max_tools == max_tools
        else:
            with pytest.raises(ValueError, match="max_tools must be positive"):
                SemanticToolSelectionTransformer(max_tools=max_tools)

    @pytest.mark.parametrize("threshold,valid", [
        (0.0, True), (0.5, True), (1.0, True), (-0.1, False), (1.1, False),
    ])
    def test_init_validates_threshold(self, threshold, valid):
        if valid:
            assert SemanticToolSelectionTransformer(similarity_threshold=threshold).threshold == threshold
        else:
            with pytest.raises(ValueError, match="similarity_threshold must be in"):
                SemanticToolSelectionTransformer(similarity_threshold=threshold)

    @patch("textile.config.get_config")
    def test_filters_tools_by_similarity(self, mock_config, sample_context, sample_tools):
        mock_embedding_model = Mock()
        mock_embedding_model.encode.side_effect = lambda text: np.random.rand(384).tolist()
        mock_config.return_value.embedding_model = mock_embedding_model
        query_embedding = np.random.rand(384).tolist()
        state = TurnState(user_message="weather forecast", user_embedding=query_embedding, tools=sample_tools)
        transformer = SemanticToolSelectionTransformer(max_tools=2, similarity_threshold=0.0)
        _, result_state = transformer.transform(sample_context, state)
        assert len(result_state.tools) <= 2

    @patch("textile.config.get_config")
    def test_respects_max_tools_limit(self, mock_config, sample_context):
        mock_embedding_model = Mock()
        mock_embedding_model.encode.side_effect = lambda text: np.random.rand(384).tolist()
        mock_config.return_value.embedding_model = mock_embedding_model
        many_tools = [{"type": "function", "function": {"name": f"tool_{i}", "description": f"Tool {i}"}} for i in range(20)]
        state = TurnState(user_message="test", user_embedding=np.random.rand(384).tolist(), tools=many_tools)
        transformer = SemanticToolSelectionTransformer(max_tools=5, similarity_threshold=0.0)
        _, result_state = transformer.transform(sample_context, state)
        assert len(result_state.tools) <= 5

    @patch("textile.config.get_config")
    def test_caches_tool_embeddings(self, mock_config, sample_context, sample_tools):
        mock_embedding_model = Mock()
        mock_embedding_model.encode.side_effect = lambda text: np.random.rand(384).tolist()
        mock_config.return_value.embedding_model = mock_embedding_model
        state = TurnState(user_message="test", user_embedding=np.random.rand(384).tolist(), tools=sample_tools)
        transformer = SemanticToolSelectionTransformer(cache_embeddings=True)
        transformer.transform(sample_context, state)
        initial_calls = mock_embedding_model.encode.call_count
        transformer.transform(sample_context, state)
        assert mock_embedding_model.encode.call_count == initial_calls

    def test_should_apply_when_tools_exceed_limit(self, sample_context):
        few_tools = [{"type": "function", "function": {"name": f"tool{i}"}} for i in range(5)]
        many_tools = [{"type": "function", "function": {"name": f"tool{i}"}} for i in range(15)]
        transformer = SemanticToolSelectionTransformer(max_tools=10)
        assert not transformer.should_apply(sample_context, TurnState("test", tools=few_tools))
        assert transformer.should_apply(sample_context, TurnState("test", tools=many_tools))

    def test_should_not_apply_without_tools(self, sample_context):
        transformer = SemanticToolSelectionTransformer()
        state = TurnState(user_message="test", tools=None)
        assert not transformer.should_apply(sample_context, state)
