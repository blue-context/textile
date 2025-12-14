"""Tests for SemanticDecayTransformer initialization and validation."""

import pytest

from textile.transformers.semantic_decay import (
    MessageType,
    SemanticDecayMetadata,
    SemanticDecayTransformer,
)


class TestMessageType:
    """Tests for MessageType enum."""

    @pytest.mark.parametrize("msg_type,expected", [
        (MessageType.SYSTEM, 1.0), (MessageType.INSTRUCTION, 0.9),
        (MessageType.FACTUAL, 0.8), (MessageType.CONVERSATIONAL, 0.6),
        (MessageType.HISTORICAL, 0.4),
    ])
    def test_get_modifier(self, msg_type, expected):
        assert msg_type.get_modifier() == expected


class TestSemanticDecayMetadata:
    """Tests for SemanticDecayMetadata."""

    def test_default_values(self):
        meta = SemanticDecayMetadata()
        assert meta.salience == 0.5
        assert meta.last_access_turn == 0
        assert meta.message_type == "conversational"

    @pytest.mark.parametrize("salience,valid", [
        (0.0, True), (0.5, True), (1.0, True), (-0.1, False), (1.1, False),
    ])
    def test_validate_salience(self, salience, valid):
        meta = SemanticDecayMetadata(salience=salience)
        if valid:
            meta.validate()
        else:
            with pytest.raises(ValueError, match="salience must be 0.0-1.0"):
                meta.validate()

    def test_validate_last_access_turn(self):
        meta = SemanticDecayMetadata(last_access_turn=-1)
        with pytest.raises(ValueError, match="last_access_turn must be >= 0"):
            meta.validate()


class TestSemanticDecayTransformerInit:
    """Tests for SemanticDecayTransformer initialization."""

    @pytest.mark.parametrize("threshold,valid", [
        (0.0, True), (0.5, True), (1.0, True), (-0.1, False), (1.1, False),
    ])
    def test_validates_threshold(self, threshold, valid):
        if valid:
            assert SemanticDecayTransformer(threshold=threshold).threshold == threshold
        else:
            with pytest.raises(ValueError, match="threshold must be between"):
                SemanticDecayTransformer(threshold=threshold)

    @pytest.mark.parametrize("semantic_threshold,valid", [
        (0.0, True), (0.5, True), (1.0, True), (-0.1, False), (1.1, False),
    ])
    def test_validates_semantic_threshold(self, semantic_threshold, valid):
        if valid:
            assert SemanticDecayTransformer(semantic_threshold=semantic_threshold).semantic_threshold == semantic_threshold
        else:
            with pytest.raises(ValueError, match="semantic_threshold must be between"):
                SemanticDecayTransformer(semantic_threshold=semantic_threshold)

    @pytest.mark.parametrize("semantic_weight,temporal_weight,valid", [
        (0.6, 0.4, True), (0.5, 0.5, True), (1.0, 0.0, True),
        (0.7, 0.2, False), (0.5, 0.4, False),
    ])
    def test_validates_weights_sum_to_one(self, semantic_weight, temporal_weight, valid):
        if valid:
            transformer = SemanticDecayTransformer(semantic_weight=semantic_weight, temporal_weight=temporal_weight)
            assert transformer.semantic_weight == semantic_weight
        else:
            with pytest.raises(ValueError, match="must sum to 1.0"):
                SemanticDecayTransformer(semantic_weight=semantic_weight, temporal_weight=temporal_weight)

    def test_validates_semantic_decay_power(self):
        with pytest.raises(ValueError, match="semantic_decay_power must be >= 1.0"):
            SemanticDecayTransformer(semantic_decay_power=0.5)

    def test_validates_half_life_turns(self):
        with pytest.raises(ValueError, match="half_life_turns must be > 0"):
            SemanticDecayTransformer(half_life_turns=0)
