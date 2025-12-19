"""Simplest Possible Transformer

This is the minimal example to understand the transformer pattern.
It filters out all user messages (obviously not useful, but clear).

Start here to understand:
1. Inherit from ContextTransformer
2. Implement transform() method
3. Return new context and state
4. Optionally implement should_apply()
"""

from textile.core.context_window import ContextWindow
from textile.core.turn_state import TurnState
from textile.transformers.base import ContextTransformer


class SimpleFilterTransformer(ContextTransformer):
    """Remove all user messages.

    This is intentionally simple (and not useful) to demonstrate
    the basic pattern without complexity.
    """

    def transform(
        self,
        context: ContextWindow,
        state: TurnState,
    ) -> tuple[ContextWindow, TurnState]:
        """Filter out user messages.

        This is the core method every transformer must implement.

        Args:
            context: The current conversation context
            state: The current turn state

        Returns:
            A tuple of (modified_context, state)
            Note: We return state unchanged (transformers don't have to modify it)
        """
        # Remove all user messages from the context
        user_message_ids = [
            msg.id for msg in context.messages if msg.role == "user"
        ]

        for msg_id in user_message_ids:
            context.remove_message(msg_id)

        # Return the modified context and unchanged state
        return context, state

    def should_apply(self, context: ContextWindow, state: TurnState) -> bool:
        """Decide if this transformer should run.

        This is optional but recommended for performance.
        Only run if there are user messages to filter.

        Args:
            context: The current conversation context
            state: The current turn state

        Returns:
            True if transformer should run, False to skip
        """
        return any(msg.role == "user" for msg in context.messages)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    import textile

    # Configure textile to use our simple transformer
    textile.configure(
        transformers=[SimpleFilterTransformer()]
    )

    # This completion will have all user messages filtered out
    # (Obviously broken, but demonstrates the mechanics)
    response = textile.completion(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "This will be removed!"},
            {"role": "assistant", "content": "I'm here."},
            {"role": "user", "content": "This will also be removed!"},
        ]
    )

    # The LLM only sees: system + assistant messages
    # User messages were filtered out by our transformer
