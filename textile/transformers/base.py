"""Abstract base class for context transformers."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from textile.core.context_window import ContextWindow
from textile.core.turn_state import TurnState

if TYPE_CHECKING:
    from textile.core.response_pattern import OnPattern


class ContextTransformer(ABC):
    """Abstract base for context transformations.

    Lifecycle:
    1. should_apply() - Gate transformer execution (optional)
    2. transform() - Modify context and/or create new state
    3. on_response() - Register response patterns (optional)

    State: Context modified in-place, state immutable (use dataclasses.replace())
    Pipeline: Execute in order, output of N becomes input to N+1
    """

    @abstractmethod
    def transform(
        self,
        context: ContextWindow,
        state: TurnState,
    ) -> tuple[ContextWindow, TurnState]:
        """Apply transformation to context and/or state."""
        pass

    def should_apply(self, context: ContextWindow, state: TurnState) -> bool:
        """Determine if transformer should run."""
        return True

    def on_response(self, state: TurnState) -> list["OnPattern"] | None:
        """Register patterns to transform LLM response."""
        return None
