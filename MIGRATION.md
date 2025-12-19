# Migration Guide: v0.3.x → v0.5.0

## Overview

Textile v0.5.0 represents an architectural shift from "middleware with transformers" to **"pure transformation infrastructure"**.

The built-in transformers are moving out of the core package and into `examples/` as **reference implementations**.

## Why This Change?

**Philosophy**: Transformers are highly domain-specific. The reference implementations are teaching examples, not production solutions. You should build transformers tailored to your use case.

**Benefits**:
- ✅ Clearer positioning: Infrastructure vs. implementations
- ✅ Lean core package (easier to maintain)
- ✅ Better learning (examples are heavily documented)
- ✅ Your code, your control (customize transformers for your domain)
- ✅ No confusion about "production-ready"

## Migration Timeline

| Version | Status | Action |
|---------|--------|--------|
| **v0.3.x** | Current | Built-in transformers work normally |
| **v0.4.0** | Deprecation | Transformers work but show warnings |
| **v0.5.0** | Breaking | Transformers removed from package |

## How to Migrate

### Step 1: Identify Your Transformers (v0.3.x → v0.4.0)

**Before (v0.3.x)**:
```python
from textile.transformers import DecayTransformer, SemanticPruningTransformer

textile.configure(
    transformers=[
        DecayTransformer(half_life_turns=5),
        SemanticPruningTransformer(similarity_threshold=0.3),
    ]
)
```

**After upgrading to v0.4.0**: This code still works but shows deprecation warnings:
```
DeprecationWarning: DecayTransformer is deprecated and will be removed in v0.5.0.
Copy from examples/reference_transformers/ to your project instead.
```

### Step 2: Copy Reference Transformers to Your Project

Browse `examples/reference_transformers/` and copy the transformers you're using:

```bash
# Create transformers directory in your project
mkdir -p my_app/transformers

# Copy the transformers you're using
cp examples/reference_transformers/temporal/decay.py my_app/transformers/
cp examples/reference_transformers/semantic/pruning.py my_app/transformers/

# Optional: Copy __init__.py for cleaner imports
touch my_app/transformers/__init__.py
```

### Step 3: Update Your Imports

**Before (v0.3.x/v0.4.0)**:
```python
from textile.transformers import DecayTransformer, SemanticPruningTransformer
```

**After (v0.5.0)**:
```python
from my_app.transformers.decay import DecayTransformer
from my_app.transformers.pruning import SemanticPruningTransformer
```

### Step 4: Customize for Your Use Case

Now that transformers are YOUR code, customize them!

**Example: Adjust decay formula**:
```python
# In my_app/transformers/decay.py
class DecayTransformer(ContextTransformer):
    def transform(self, context, state):
        for msg in context.messages:
            age_turns = current_turn - msg.turn_index

            # Change: Linear decay instead of exponential
            decay_factor = max(0, 1 - (age_turns / self.max_age))
            # Old: decay_factor = 0.5 ** (age_turns / self.half_life)

            msg.metadata.prominence *= decay_factor
        # ... rest of logic
```

## Transformer Mappings

| Old Import (v0.3.x) | New Location (v0.5.0) |
|---------------------|------------------------|
| `textile.transformers.DecayTransformer` | `examples/reference_transformers/temporal/decay.py` |
| `textile.transformers.SemanticPruningTransformer` | `examples/reference_transformers/semantic/pruning.py` |
| `textile.transformers.SemanticDecayTransformer` | `examples/reference_transformers/semantic/semantic_decay.py` |
| `textile.transformers.SemanticToolSelectionTransformer` | `examples/reference_transformers/semantic/tool_selection.py` |

**Core classes remain unchanged**:
- ✅ `textile.transformers.ContextTransformer` (protocol/base class)
- ✅ `textile.transformers.TransformationPipeline` (orchestration)
- ✅ All core APIs (`textile.completion`, `textile.configure`, etc.)

## Complete Migration Example

### Before (v0.3.x)

```python
# app.py
import textile
from textile.transformers import (
    DecayTransformer,
    SemanticPruningTransformer,
)
from textile.embeddings import Embedding

textile.configure(
    embedding_model=Embedding("text-embedding-3-small"),
    transformers=[
        DecayTransformer(half_life_turns=5, threshold=0.1),
        SemanticPruningTransformer(similarity_threshold=0.3),
    ]
)

response = textile.completion(
    model="gpt-4",
    messages=[...]
)
```

### After (v0.5.0)

```python
# app.py
import textile
from textile.embeddings import Embedding

# Import from YOUR code
from my_app.transformers.decay import DecayTransformer
from my_app.transformers.pruning import SemanticPruningTransformer

textile.configure(
    embedding_model=Embedding("text-embedding-3-small"),
    transformers=[
        DecayTransformer(half_life_turns=5, threshold=0.1),
        SemanticPruningTransformer(similarity_threshold=0.3),
    ]
)

response = textile.completion(
    model="gpt-4",
    messages=[...]
)
```

**Project structure**:
```
my_app/
├── __init__.py
├── app.py                     # Your application code
└── transformers/              # Your transformers (copied from examples)
    ├── __init__.py
    ├── decay.py               # Copied and customized
    └── pruning.py             # Copied and customized
```

## Testing After Migration

Verify your migration with these tests:

```python
# test_migration.py
import pytest
import warnings

def test_no_deprecation_warnings():
    """Ensure no deprecation warnings after migration."""
    import textile
    from my_app.transformers.decay import DecayTransformer  # Your code

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)

        # This should NOT raise DeprecationWarning
        textile.configure(transformers=[DecayTransformer()])

def test_transformers_work():
    """Verify transformers still function correctly."""
    import textile
    from my_app.transformers.decay import DecayTransformer

    textile.configure(transformers=[DecayTransformer(half_life_turns=5)])

    response = textile.completion(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"}
        ]
    )

    assert response.choices[0].message.content
```

## What's NOT Changing

These remain the same:

✅ **Core Package APIs**:
- `textile.completion()`
- `textile.acompletion()`
- `textile.configure()`
- All LiteLLM compatibility functions

✅ **Core Classes**:
- `ContextTransformer` (protocol)
- `TransformationPipeline`
- `ContextWindow`, `Message`, `TurnState`
- `Embedding` abstraction

✅ **Infrastructure**:
- Async/await support
- Streaming support
- LiteLLM compatibility
- Zero API changes philosophy

## New Features (v0.4.0+)

While migrating, take advantage of new infrastructure:

### 1. Testing Utilities
```python
from textile.utils.testing import TransformerTestCase, create_context

class TestMyTransformer(TransformerTestCase):
    def test_removes_old_messages(self):
        context, state = self.create_test_context(
            messages=[...],
            current_turn=10
        )
        # ... test your transformer
```

### 2. Caching Infrastructure
```python
from textile.cache import MemoryCache

class MyTransformer(ContextTransformer):
    def __init__(self):
        self.cache = MemoryCache()

    def transform(self, context, state):
        # Use cache for expensive operations
        if self.cache.exists(key):
            return self.cache.get(key)
        # ... compute ...
        self.cache.set(key, result)
```

### 3. Observability Hooks
```python
from textile.hooks import MetricsHook

hook = MetricsHook()

def log_metrics(metrics):
    print(f"{metrics.transformer_name}: {metrics.duration_ms:.2f}ms, "
          f"removed {metrics.messages_removed} messages")

hook.register_callback(log_metrics)

# Use hook in your transformer
# ... metrics are automatically collected
```

## Troubleshooting

### Problem: Import errors after upgrading

**Error**:
```python
ImportError: cannot import name 'DecayTransformer' from 'textile.transformers'
```

**Solution**: You upgraded to v0.5.0 without migrating. Copy transformers from `examples/`:
```bash
cp examples/reference_transformers/temporal/decay.py my_app/transformers/
```

### Problem: Deprecation warnings in v0.4.0

**Warning**:
```
DeprecationWarning: DecayTransformer is deprecated...
```

**Solution**: This is expected in v0.4.0. Follow migration steps above to silence warnings.

### Problem: Tests breaking after migration

**Error**: Tests fail with `DeprecationWarning` treated as errors

**Solution**: Update test imports:
```python
# Before
from textile.transformers import DecayTransformer

# After
from my_app.transformers.decay import DecayTransformer
```

## Getting Help

- **Issues**: https://github.com/blue-context/textile/issues
- **Discussions**: https://github.com/blue-context/textile/discussions
- **Examples**: https://github.com/blue-context/textile/tree/main/examples

## Summary Checklist

- [ ] Upgrade to v0.4.0 to see deprecation warnings
- [ ] Identify which transformers you're using
- [ ] Copy transformers from `examples/reference_transformers/` to your project
- [ ] Update imports in your code
- [ ] Customize transformers for your use case
- [ ] Run tests to verify migration
- [ ] Optional: Explore new infrastructure features (caching, hooks, testing)

**Remember**: Transformers are now YOUR code. Make them work for YOUR use case!
