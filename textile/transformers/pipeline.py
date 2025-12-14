"""Transformation pipeline for orchestrating multiple transformers."""

from textile.core.context_window import ContextWindow
from textile.core.turn_state import TurnState
from textile.transformers.base import ContextTransformer


class TransformationPipeline:
    """Execute transformers sequentially with state threading.

    Sequential execution with in-place context modification and immutable state.
    Debug mode captures trace snapshots.
    """

    def __init__(self, transformers: list[ContextTransformer], debug: bool = False) -> None:
        """Initialize pipeline."""
        self.transformers = transformers
        self.debug = debug
        self.trace: list[dict] = []

    def apply(
        self,
        context: ContextWindow,
        state: TurnState,
    ) -> tuple[ContextWindow, TurnState]:
        """Apply transformers sequentially, threading state through."""
        current_state = state

        if self.debug:
            self.trace = []
            self.trace.append(
                {
                    "step": "initial",
                    "transformer": None,
                    "messages_before": len(context.messages),
                    "messages": [{"role": m.role, "content": m.content} for m in context.messages],
                }
            )

        for i, transformer in enumerate(self.transformers):
            if transformer.should_apply(context, current_state):
                messages_before = len(context.messages)
                context, current_state = transformer.transform(context, current_state)

                if self.debug:
                    self.trace.append(
                        {
                            "step": i + 1,
                            "transformer": transformer.__class__.__name__,
                            "messages_before": messages_before,
                            "messages_after": len(context.messages),
                            "messages_removed": messages_before - len(context.messages),
                            "messages": [
                                {"role": m.role, "content": m.content} for m in context.messages
                            ],
                        }
                    )

        return context, current_state

    def add_transformer(self, transformer: ContextTransformer) -> None:
        """Add transformer to pipeline end."""
        self.transformers.append(transformer)

    def remove_transformer(self, transformer_type: type) -> bool:
        """Remove first transformer of given type."""
        for i, t in enumerate(self.transformers):
            if isinstance(t, transformer_type):
                self.transformers.pop(i)
                return True
        return False
