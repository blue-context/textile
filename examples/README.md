# Building Custom Transformers

**Textile is transformation infrastructure.** The built-in transformers in this directory are **reference implementations** - teaching examples, not production solutions.

Your transformers should be tailored to your specific use case, domain, and requirements.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [Transformation Patterns](#transformation-patterns)
4. [Best Practices](#best-practices)
5. [Testing Your Transformers](#testing-your-transformers)
6. [Performance Guidelines](#performance-guidelines)
7. [Common Pitfalls](#common-pitfalls)

---

## Quick Start

### 1. Start with the Simplest Example

```bash
# Read the simplest possible transformer
cat examples/quickstart/simple_filter.py
```

This shows the bare minimum: inherit from `ContextTransformer`, implement `transform()`.

### 2. Copy a Reference Implementation

Browse `reference_transformers/` and find one that matches your pattern:

```bash
# Copy temporal decay as a starting point
cp examples/reference_transformers/temporal/decay.py my_app/transformers/

# Or semantic pruning
cp examples/reference_transformers/semantic/pruning.py my_app/transformers/
```

### 3. Customize for Your Use Case

The reference implementations are **templates**. Modify them:
- Adjust formulas (linear instead of exponential decay)
- Add domain-specific logic (user activity, importance scores)
- Combine patterns (temporal + semantic + your domain knowledge)

### 4. Test Thoroughly

```python
# Use the testing utilities (see Testing section)
from textile.utils.testing import assert_messages_removed, TransformerTestCase

class TestMyTransformer(TransformerTestCase):
    def test_removes_old_messages(self):
        result = self.apply_transformer(
            MyTransformer(),
            messages=[...],
            expected_removed=2
        )
```

---

## Core Concepts

### The Transformer Protocol

Every transformer must implement this interface:

```python
from textile.transformers.base import ContextTransformer
from textile.core.context_window import ContextWindow
from textile.core.turn_state import TurnState

class MyTransformer(ContextTransformer):
    def transform(
        self,
        context: ContextWindow,
        state: TurnState
    ) -> tuple[ContextWindow, TurnState]:
        """Transform the context and/or state.

        Args:
            context: Current conversation context
            state: Current turn state

        Returns:
            Tuple of (modified_context, modified_state)
        """
        # Your logic here
        return context, state

    def should_apply(
        self,
        context: ContextWindow,
        state: TurnState
    ) -> bool:
        """Optional: Decide if transformer should run.

        Returns:
            True to run transform(), False to skip
        """
        return len(context.messages) > 1
```

### Key Objects

**ContextWindow**: The conversation history
```python
context.messages          # List[Message] - all messages
context.add_message(msg)  # Add a message
context.remove_message(id) # Remove by ID
context.get_message_by_id(id) # Retrieve message
```

**TurnState**: Metadata about current turn
```python
state.turn_index       # int - current turn number
state.user_embedding   # Current query embedding
state.tools            # Available tools/functions
state.metadata         # dict - custom metadata
```

**Message**: Individual message
```python
msg.id          # Unique identifier
msg.role        # "system", "user", "assistant"
msg.content     # Message text
msg.turn_index  # When message was created
msg.embedding   # Optional embedding
msg.metadata    # Metadata object
```

---

## Transformation Patterns

### Pattern 1: Filtering (Remove Messages)

**Use for**: Removing irrelevant, old, or low-value messages

```python
class FilteringTransformer(ContextTransformer):
    def transform(self, context, state):
        # Identify messages to remove
        to_remove = [
            msg.id for msg in context.messages
            if self._should_remove(msg, state)
        ]

        # Remove them
        for msg_id in to_remove:
            context.remove_message(msg_id)

        return context, state

    def _should_remove(self, msg, state):
        # Your filtering logic
        return msg.metadata.prominence < 0.1
```

**Examples**: `decay.py`, `pruning.py`

### Pattern 2: Modification (Change Content)

**Use for**: Rewriting, summarizing, or enhancing messages

```python
class ModificationTransformer(ContextTransformer):
    def transform(self, context, state):
        for msg in context.messages:
            if self._should_modify(msg):
                msg.content = self._modify_content(msg.content)

        return context, state

    def _modify_content(self, content):
        # Your modification logic
        return content.upper()  # Example: capitalize
```

**Examples**: Summarization, truncation, formatting

### Pattern 3: State Transformation (Modify Tools/Metadata)

**Use for**: Filtering tools, updating state, managing resources

```python
class StateTransformer(ContextTransformer):
    def transform(self, context, state):
        from dataclasses import replace

        # Filter tools
        filtered_tools = [
            tool for tool in state.tools
            if self._is_relevant(tool, state)
        ]

        # Return new state (immutable)
        new_state = replace(state, tools=filtered_tools)
        return context, new_state
```

**Examples**: `tool_selection.py`

### Pattern 4: Semantic (Using Embeddings)

**Use for**: Relevance-based operations requiring understanding

```python
class SemanticTransformer(ContextTransformer):
    def transform(self, context, state):
        from textile.config import get_config
        from textile.utils.similarity import cosine_similarity

        # Get embedding model from config
        config = get_config()
        model = config.embedding_model

        # Compute similarity
        for msg in context.messages:
            if msg.embedding is not None:
                similarity = cosine_similarity(
                    state.user_embedding,
                    msg.embedding
                )
                # Use similarity in your logic
                msg.metadata._set_raw("similarity", similarity)

        return context, state
```

**Examples**: `pruning.py`, `semantic_decay.py`

### Pattern 5: Hybrid (Combining Multiple Factors)

**Use for**: Complex scoring combining time, relevance, importance

```python
class HybridTransformer(ContextTransformer):
    def __init__(self, temporal_weight=0.5, semantic_weight=0.5):
        self.temporal_weight = temporal_weight
        self.semantic_weight = semantic_weight

    def transform(self, context, state):
        for msg in context.messages:
            # Compute multiple scores
            temporal_score = self._compute_temporal_score(msg, state)
            semantic_score = self._compute_semantic_score(msg, state)

            # Combine them
            final_score = (
                self.temporal_weight * temporal_score +
                self.semantic_weight * semantic_score
            )

            msg.metadata.prominence = final_score

        # Filter based on combined score
        # ...

        return context, state
```

**Examples**: `semantic_decay.py`

---

## Best Practices

### ✅ DO

1. **Keep transformers stateless**
   ```python
   # Good: No instance state modified
   def transform(self, context, state):
       threshold = self.threshold  # Read-only config
       # ...
   ```

   ```python
   # Bad: Modifying instance state
   def transform(self, context, state):
       self.message_count += len(context.messages)  # Stateful!
       # ...
   ```

2. **Return new objects for immutability**
   ```python
   # Good: Create new state
   new_state = replace(state, tools=filtered_tools)
   return context, new_state
   ```

   ```python
   # Bad: Mutate input state
   state.tools = filtered_tools  # Don't do this!
   return context, state
   ```

3. **Handle edge cases gracefully**
   ```python
   # Always ensure at least one non-system message
   if not any(msg.role != "system" for msg in context.messages):
       # Keep the best message
       best = max(messages, key=lambda m: m.metadata.prominence)
       messages_to_keep.add(best.id)
   ```

4. **Use `should_apply()` to skip unnecessary work**
   ```python
   def should_apply(self, context, state):
       # Only run if there's something to do
       return len(context.messages) > 5
   ```

5. **Log extensively for debugging**
   ```python
   import logging
   logger = logging.getLogger(__name__)

   def transform(self, context, state):
       logger.debug(f"Before: {len(context.messages)} messages")
       # ... transform ...
       logger.debug(f"After: {len(context.messages)} messages, removed {removed}")
   ```

6. **Store metrics in metadata**
   ```python
   # Track what the transformer did
   state.metadata["messages_removed"] = removed_count
   state.metadata["avg_similarity"] = avg_similarity
   ```

### ❌ DON'T

1. **Don't mutate input parameters** (except Message metadata - that's OK)
2. **Don't make API calls in `transform()`** (use async helpers if needed)
3. **Don't assume message order** (messages may be reordered by other transformers)
4. **Don't silently fail** (log warnings, raise errors for invalid config)
5. **Don't remove all messages** (always keep system + at least one message)
6. **Don't forget to validate config** (check thresholds, weights in `__init__`)

---

## Testing Your Transformers

### Basic Test Pattern

```python
import pytest
from textile.core.context_window import ContextWindow
from textile.core.message import Message
from textile.core.turn_state import TurnState

def test_my_transformer():
    # Setup
    messages = [
        Message.from_dict({"role": "system", "content": "You are helpful"}),
        Message.from_dict({"role": "user", "content": "Hello"}),
        Message.from_dict({"role": "assistant", "content": "Hi"}),
    ]
    context = ContextWindow(messages=messages)
    state = TurnState(turn_index=2)

    # Execute
    transformer = MyTransformer(threshold=0.5)
    new_context, new_state = transformer.transform(context, state)

    # Assert
    assert len(new_context.messages) == 2
    assert new_context.messages[0].role == "system"  # System preserved
```

### Using Test Utilities

```python
from textile.utils.testing import TransformerTestCase, assert_messages_removed

class TestMyTransformer(TransformerTestCase):
    def test_removes_old_messages(self):
        # Use helper methods
        context, state = self.create_context(
            messages=[
                {"role": "user", "content": "Old", "turn_index": 0},
                {"role": "user", "content": "New", "turn_index": 10},
            ],
            current_turn=10
        )

        transformer = MyTransformer(max_age=5)
        result_context, _ = transformer.transform(context, state)

        # Assert using utility
        assert_messages_removed(result_context, expected=1)
        assert result_context.messages[0].content == "New"
```

---

## Performance Guidelines

### Computational Complexity

| Operation | Target | Notes |
|-----------|--------|-------|
| Non-semantic transform | < 5ms | Simple filtering, decay |
| Semantic transform | < 50ms | With embedding cache |
| First-time semantic | < 200ms | Cold start, no cache |

### Optimization Strategies

1. **Cache expensive operations**
   ```python
   class CachedTransformer(ContextTransformer):
       def __init__(self):
           self._cache = {}

       def transform(self, context, state):
           # Check cache first
           if key in self._cache:
               return self._cache[key]
           # ... compute ...
           self._cache[key] = result
           return result
   ```

2. **Early exit in `should_apply()`**
   ```python
   def should_apply(self, context, state):
       # Fast checks first
       if len(context.messages) < 2:
           return False
       if not state.user_embedding:
           return False
       return True
   ```

3. **Batch operations**
   ```python
   # Good: Batch similarity computation
   similarities = compute_all_similarities(messages, query)

   # Bad: One at a time
   for msg in messages:
       similarity = compute_similarity(msg, query)
   ```

---

## Common Pitfalls

### 1. Topic Thrashing (Semantic Pruning)

**Problem**: User changes topic → all history removed → context lost

**Solution**: Combine with temporal decay or keep minimum recent messages

```python
# Always keep N recent messages regardless of similarity
recent_messages = sorted(messages, key=lambda m: m.turn_index)[-5:]
for msg in recent_messages:
    keep_ids.add(msg.id)
```

### 2. Over-Pruning

**Problem**: Threshold too aggressive → removes too much → LLM has no context

**Solution**: Always enforce minimum message count

```python
if len(messages_to_keep) < self.min_messages:
    # Keep the best messages up to min_messages
    best_messages = sorted(messages, key=score_func)[:self.min_messages]
    messages_to_keep.update(m.id for m in best_messages)
```

### 3. Embedding Cold Start

**Problem**: First call is slow (no cached embeddings)

**Solution**: Pre-compute embeddings for static content

```python
# Pre-compute system message embeddings
self.system_embeddings = {
    msg.id: model.encode(msg.content)
    for msg in context.messages
    if msg.role == "system"
}
```

### 4. State Mutation

**Problem**: Mutating shared state → unexpected behavior in pipelines

**Solution**: Always use `dataclasses.replace()`

```python
from dataclasses import replace

# Good
new_state = replace(state, tools=filtered_tools)
return context, new_state

# Bad
state.tools = filtered_tools  # Mutates input!
return context, state
```

---

## Reference Implementations

### By Pattern

| Pattern | Example | Use Case |
|---------|---------|----------|
| Temporal Filtering | `temporal/decay.py` | Age-based pruning |
| Semantic Filtering | `semantic/pruning.py` | Relevance filtering |
| Hybrid | `semantic/semantic_decay.py` | Time + relevance |
| State Transform | `semantic/tool_selection.py` | Tool filtering |
| Simple Filtering | `quickstart/simple_filter.py` | Learning basics |

### Directory Structure

```
examples/
├── README.md                          # This file
├── quickstart/
│   └── simple_filter.py              # Simplest example
├── reference_transformers/
│   ├── temporal/
│   │   ├── decay.py                  # Exponential decay
│   │   └── README.md                 # When to use, customization
│   ├── semantic/
│   │   ├── pruning.py                # Semantic filtering
│   │   ├── semantic_decay.py         # Hybrid approach
│   │   ├── tool_selection.py         # Tool filtering
│   │   └── README.md
│   ├── filtering/
│   │   ├── keyword_filter.py         # Simple keyword filtering
│   │   ├── role_filter.py            # Filter by role
│   │   └── README.md
│   └── compression/
│       ├── message_merger.py         # Combine sequential messages
│       └── README.md
└── use_cases/                        # Domain-specific examples
    ├── chatbot/
    ├── rag/
    └── agents/
```

---

## Next Steps

1. **Start Simple**: Begin with `quickstart/simple_filter.py`
2. **Copy a Reference**: Find the pattern closest to your needs
3. **Customize**: Modify for your domain and requirements
4. **Test**: Write comprehensive tests
5. **Iterate**: Measure performance, adjust parameters
6. **Share**: Consider open-sourcing your transformers for others!

---

## Getting Help

- **Issues**: https://github.com/blue-context/textile/issues
- **Discussions**: https://github.com/blue-context/textile/discussions
- **Examples**: Browse this directory for more patterns

Remember: **Transformers are YOUR code.** Make them work for your use case!
