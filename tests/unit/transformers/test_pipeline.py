"""Tests for TransformationPipeline."""

import pytest

from textile.core.context_window import ContextWindow
from textile.core.message import Message
from textile.core.turn_state import TurnState
from textile.transformers.base import ContextTransformer
from textile.transformers.decay import DecayTransformer
from textile.transformers.pipeline import TransformationPipeline


class TestTransformer(ContextTransformer):
    """Test transformer that removes user messages."""

    def transform(self, context, state):
        context.messages = [m for m in context.messages if m.role != "user"]
        return context, state


class ConditionalTransformer(ContextTransformer):
    """Transformer that only applies if state has metadata flag."""

    def should_apply(self, context, state):
        return state.metadata.get("apply_conditional", False)

    def transform(self, context, state):
        context.messages.append(Message(role="assistant", content="Added"))
        return context, state


class TestTransformationPipeline:
    """Tests for transformation pipeline."""

    def test_executes_transformers_sequentially(self, sample_context, sample_state):
        initial_count = len(sample_context.messages)
        pipeline = TransformationPipeline([TestTransformer()])
        context, _ = pipeline.apply(sample_context, sample_state)
        user_msgs = [m for m in context.messages if m.role == "user"]
        assert len(user_msgs) == 0
        assert len(context.messages) < initial_count

    def test_threads_state_through_transformers(self, sample_context):
        pipeline = TransformationPipeline([DecayTransformer()])
        state = TurnState(user_message="Test", turn_index=10)
        _, result_state = pipeline.apply(sample_context, state)
        assert result_state.turn_index == 10

    def test_respects_should_apply_guard(self, sample_context):
        state = TurnState(user_message="Test", metadata={"apply_conditional": False})
        pipeline = TransformationPipeline([ConditionalTransformer()])
        initial_count = len(sample_context.messages)
        context, _ = pipeline.apply(sample_context, state)
        assert len(context.messages) == initial_count

    def test_debug_mode_captures_trace(self, sample_context, sample_state):
        pipeline = TransformationPipeline([TestTransformer()], debug=True)
        pipeline.apply(sample_context, sample_state)
        assert len(pipeline.trace) == 2
        assert pipeline.trace[0]["step"] == "initial"
        assert pipeline.trace[1]["transformer"] == "TestTransformer"
        assert "messages_removed" in pipeline.trace[1]

    def test_add_transformer(self, sample_context, sample_state):
        pipeline = TransformationPipeline([])
        assert len(pipeline.transformers) == 0
        pipeline.add_transformer(TestTransformer())
        assert len(pipeline.transformers) == 1
        context, _ = pipeline.apply(sample_context, sample_state)
        assert len([m for m in context.messages if m.role == "user"]) == 0

    def test_remove_transformer(self):
        pipeline = TransformationPipeline([DecayTransformer(), TestTransformer()])
        removed = pipeline.remove_transformer(DecayTransformer)
        assert removed is True
        assert len(pipeline.transformers) == 1
        assert isinstance(pipeline.transformers[0], TestTransformer)

    def test_remove_transformer_not_found(self):
        pipeline = TransformationPipeline([TestTransformer()])
        removed = pipeline.remove_transformer(DecayTransformer)
        assert removed is False
        assert len(pipeline.transformers) == 1

    def test_empty_pipeline_returns_unchanged(self, sample_context, sample_state):
        pipeline = TransformationPipeline([])
        original_count = len(sample_context.messages)
        context, state = pipeline.apply(sample_context, sample_state)
        assert len(context.messages) == original_count
        assert state == sample_state
