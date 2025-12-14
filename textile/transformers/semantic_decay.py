"""Semantic decay transformer combining temporal and semantic relevance."""

import math
from dataclasses import dataclass
from enum import Enum

from textile.core.context_window import ContextWindow
from textile.core.message import Message
from textile.core.metadata import DataclassMetadata
from textile.core.turn_state import TurnState
from textile.transformers.base import ContextTransformer
from textile.utils.similarity import cosine_similarity

DEFAULT_SALIENCE = 0.5
DEFAULT_PROMINENCE = 1.0
DEFAULT_SEMANTIC_WEIGHT = 0.6
DEFAULT_TEMPORAL_WEIGHT = 0.4

TYPE_MODIFIER_SYSTEM = 1.0
TYPE_MODIFIER_INSTRUCTION = 0.9
TYPE_MODIFIER_FACTUAL = 0.8
TYPE_MODIFIER_CONVERSATIONAL = 0.6
TYPE_MODIFIER_HISTORICAL = 0.4


class MessageType(str, Enum):
    """Message type classification for decay calculations."""

    SYSTEM = "system"
    INSTRUCTION = "instruction"
    FACTUAL = "factual"
    CONVERSATIONAL = "conversational"
    HISTORICAL = "historical"

    def get_modifier(self) -> float:
        """Get decay modifier for message type."""
        return {
            MessageType.SYSTEM: TYPE_MODIFIER_SYSTEM,
            MessageType.INSTRUCTION: TYPE_MODIFIER_INSTRUCTION,
            MessageType.FACTUAL: TYPE_MODIFIER_FACTUAL,
            MessageType.CONVERSATIONAL: TYPE_MODIFIER_CONVERSATIONAL,
            MessageType.HISTORICAL: TYPE_MODIFIER_HISTORICAL,
        }[self]


@dataclass
class SemanticDecayMetadata(DataclassMetadata):
    """Metadata for semantic decay transformer."""

    salience: float = DEFAULT_SALIENCE
    last_access_turn: int = 0
    message_type: str = MessageType.CONVERSATIONAL.value

    def validate(self) -> None:
        """Validate metadata values."""
        if not 0.0 <= self.salience <= 1.0:
            raise ValueError(f"salience must be 0.0-1.0, got {self.salience}")
        if self.last_access_turn < 0:
            raise ValueError(f"last_access_turn must be >= 0, got {self.last_access_turn}")


class SemanticDecayTransformer(ContextTransformer):
    """Combine temporal decay with semantic relevance for pruning.

    Removes messages that are old AND off-topic. Keeps old BUT on-topic.
    Relevance: R0 * m_type * [w_sem*D_sem + w_temp*D_temp] * D_sal * w_rec
    System messages always preserved.
    """

    def __init__(
        self,
        half_life_turns: int = 4,
        threshold: float = 0.1,
        semantic_threshold: float = 0.3,
        semantic_decay_power: float = 1.5,
        semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
        temporal_weight: float = DEFAULT_TEMPORAL_WEIGHT,
        salience_decay: float = 0.2,
        recency_multiplier: float = 0.3,
        recency_threshold: int = 10,
    ) -> None:
        """Initialize semantic decay transformer.

        Args:
            half_life_turns: Turns for temporal decay to 50%
            threshold: Minimum relevance to keep (0.0-1.0)
            semantic_threshold: Minimum similarity to keep (0.0-1.0)
            semantic_decay_power: Semantic decay power (higher=stricter)
            semantic_weight: Semantic component weight (0.0-1.0)
            temporal_weight: Temporal component weight (0.0-1.0)
            salience_decay: Salience boost decay rate
            recency_multiplier: Recent access boost multiplier
            recency_threshold: Turns for recency boost

        Raises:
            ValueError: If parameters invalid or weights don't sum to 1.0
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be between 0.0 and 1.0, got {threshold}")

        if not 0.0 <= semantic_threshold <= 1.0:
            raise ValueError(
                f"semantic_threshold must be between 0.0 and 1.0, got {semantic_threshold}"
            )

        if not 0.0 <= semantic_weight <= 1.0:
            raise ValueError(f"semantic_weight must be between 0.0 and 1.0, got {semantic_weight}")

        if not 0.0 <= temporal_weight <= 1.0:
            raise ValueError(f"temporal_weight must be between 0.0 and 1.0, got {temporal_weight}")

        if not math.isclose(semantic_weight + temporal_weight, 1.0, abs_tol=0.01):
            raise ValueError(
                f"semantic_weight and temporal_weight must sum to 1.0, "
                f"got {semantic_weight} + {temporal_weight} = {semantic_weight + temporal_weight}"
            )

        if semantic_decay_power < 1.0:
            raise ValueError(f"semantic_decay_power must be >= 1.0, got {semantic_decay_power}")

        if half_life_turns <= 0:
            raise ValueError(f"half_life_turns must be > 0, got {half_life_turns}")

        self.half_life = half_life_turns
        self.threshold = threshold
        self.semantic_threshold = semantic_threshold
        self.semantic_decay_power = semantic_decay_power
        self.semantic_weight = semantic_weight
        self.temporal_weight = temporal_weight
        self.salience_decay = salience_decay
        self.recency_multiplier = recency_multiplier
        self.recency_threshold = recency_threshold

    def transform(
        self,
        context: ContextWindow,
        state: TurnState,
    ) -> tuple[ContextWindow, TurnState]:
        """Apply semantic decay and prune irrelevant messages.

        Args:
            context: Context window
            state: Turn state

        Returns:
            Tuple of (modified context, unchanged state)
        """
        if not context.messages:
            return context, state

        current_turn = state.turn_index
        query_embedding = state.user_embedding
        has_embeddings = any(msg.embedding is not None for msg in context.messages)

        for msg in context.messages:
            decay_meta = msg.metadata.get_namespace("semantic_decay", SemanticDecayMetadata)

            if decay_meta is None:
                decay_meta = SemanticDecayMetadata(
                    salience=DEFAULT_SALIENCE,
                    last_access_turn=msg.turn_index,
                    message_type=self._infer_message_type(msg).value,
                )
                msg.metadata.set_namespace("semantic_decay", decay_meta)

        for msg in context.messages:
            self._calculate_relevance(msg, current_turn, query_embedding, has_embeddings)

        messages_to_keep = self._filter_messages(context, has_embeddings)

        filtered_messages = [msg for msg in context.messages if msg.id in messages_to_keep]
        removed_count = len(context.messages) - len(filtered_messages)
        context.messages = filtered_messages

        if removed_count > 0:
            state.metadata["semantic_decay_pruned"] = removed_count

        return context, state

    def should_apply(self, context: ContextWindow, state: TurnState) -> bool:
        """Apply only if multiple messages.

        Args:
            context: Context window
            state: Turn state

        Returns:
            True if 2+ messages
        """
        return len(context.messages) > 1

    def _infer_message_type(self, msg: Message) -> MessageType:
        """Infer message type from role and metadata.

        Args:
            msg: Message

        Returns:
            MessageType enum
        """
        if msg.role == "system":
            return MessageType.SYSTEM
        if msg.role == "user" and msg.metadata._get_raw("is_instruction"):
            return MessageType.INSTRUCTION
        if msg.metadata._get_raw("is_factual"):
            return MessageType.FACTUAL
        if msg.metadata._get_raw("is_historical"):
            return MessageType.HISTORICAL
        return MessageType.CONVERSATIONAL

    def _get_message_type(self, msg: Message) -> MessageType:
        """Get message type from metadata.

        Args:
            msg: Message

        Returns:
            MessageType from metadata or inferred
        """
        decay_meta = msg.metadata.get_namespace("semantic_decay", SemanticDecayMetadata)
        if decay_meta is not None:
            return MessageType(decay_meta.message_type)
        return self._infer_message_type(msg)

    def _calculate_relevance(
        self,
        msg: Message,
        current_turn: int,
        query_embedding,
        has_embeddings: bool,
    ) -> float:
        """Calculate message relevance score.

        Args:
            msg: Message to score
            current_turn: Current turn index
            query_embedding: Query embedding for similarity
            has_embeddings: Whether embeddings are available

        Returns:
            Relevance score combining decay, type, salience, and recency
        """
        decay_meta = msg.metadata.get_namespace("semantic_decay", SemanticDecayMetadata)
        if decay_meta is None:
            decay_meta = SemanticDecayMetadata(
                salience=DEFAULT_SALIENCE,
                last_access_turn=msg.turn_index,
                message_type=self._infer_message_type(msg).value,
            )

        age_turns = current_turn - msg.turn_index
        initial_prominence = msg.metadata.prominence
        msg_type = MessageType(decay_meta.message_type)
        type_modifier = msg_type.get_modifier()

        temporal_decay = 0.5 ** (age_turns / self.half_life)

        semantic_decay = 1.0
        similarity_score = 1.0

        if has_embeddings and msg.embedding is not None and query_embedding is not None:
            try:
                similarity_score = cosine_similarity(query_embedding, msg.embedding)
                semantic_decay = similarity_score**self.semantic_decay_power
            except (ValueError, RuntimeError):
                semantic_decay = 1.0
                similarity_score = 1.0

        combined_decay = (
            self.semantic_weight * semantic_decay + self.temporal_weight * temporal_decay
        )

        salience_boost = 1.0 + (decay_meta.salience * self.salience_decay)

        turns_since_access = current_turn - decay_meta.last_access_turn
        recency_boost = 1.0
        if turns_since_access <= self.recency_threshold:
            recency_factor = 1.0 - turns_since_access / self.recency_threshold
            recency_boost = 1.0 + self.recency_multiplier * recency_factor

        relevance = initial_prominence * type_modifier * combined_decay * salience_boost * recency_boost

        decay_meta.last_access_turn = current_turn

        msg.metadata.prominence = relevance
        msg.metadata.set_namespace("semantic_decay", decay_meta)
        msg.metadata._set_raw("relevance", relevance)

        msg.metadata._set_raw(
            "decay_components",
            {
                "R0": initial_prominence,
                "m_type": type_modifier,
                "D_semantic": semantic_decay,
                "D_temporal": temporal_decay,
                "combined_decay": combined_decay,
                "D_salience": salience_boost,
                "w_recency": recency_boost,
                "similarity": similarity_score,
                "age_turns": age_turns,
                "salience": decay_meta.salience,
            },
        )

        return relevance

    def _filter_messages(
        self,
        context: ContextWindow,
        has_embeddings: bool,
    ) -> set[str]:
        """Filter messages based on relevance and semantic thresholds.

        Args:
            context: Context window with scored messages
            has_embeddings: Whether embeddings are available

        Returns:
            Set of message IDs to keep
        """
        messages_to_keep = set()

        for msg in context.messages:
            if msg.role == "system":
                messages_to_keep.add(msg.id)

        non_system_above_threshold = []

        for msg in context.messages:
            if msg.role == "system":
                continue

            relevance = msg.metadata._get_raw("relevance") or 1.0
            decay_components = msg.metadata._get_raw("decay_components") or {}
            similarity = decay_components.get("similarity", 1.0)

            if relevance >= self.threshold:
                if not has_embeddings or similarity >= self.semantic_threshold:
                    non_system_above_threshold.append(msg)
                    messages_to_keep.add(msg.id)

        non_system_messages = [msg for msg in context.messages if msg.role != "system"]
        if non_system_messages and not non_system_above_threshold:
            best_message = max(
                non_system_messages, key=lambda m: m.metadata._get_raw("relevance") or 1.0
            )
            messages_to_keep.add(best_message.id)

        return messages_to_keep
