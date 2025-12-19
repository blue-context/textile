# Textile

**Transformation infrastructure for LLM applications.**

Textile provides the pipeline, protocols, and infrastructure for building custom message transformers. You bring the domain-specific logic.

[![PyPI](https://img.shields.io/pypi/v/textile-llm)](https://pypi.org/project/textile-llm/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## ⚠️ v0.5.0 Breaking Change

**Built-in transformers have been removed.**

If upgrading from v0.3.x or v0.4.0, see [MIGRATION.md](MIGRATION.md) for the upgrade guide.

All transformers are now reference implementations in `examples/reference_transformers/` - copy them to your project and customize for your domain.

## What is Textile?

Textile is a **drop-in replacement** for `litellm.completion()` that lets you transform messages before they reach the LLM - **with zero API changes**.

It's pure infrastructure: transformation pipelines, protocols, caching, observability. **Not** a transformer library.

## Philosophy

**Transformers are domain-specific code that YOU write.**

The transformers in `examples/reference_transformers/` are teaching examples, not production solutions. Copy them to your project and customize for your use case.

Textile provides:
- ✅ Transformation pipeline infrastructure
- ✅ `ContextTransformer` protocol
- ✅ Testing utilities for custom transformers
- ✅ Caching for expensive operations
- ✅ Observability hooks
- ✅ Zero-change LiteLLM compatibility

You provide:
- Your domain logic (age decay? relevance filtering? custom scoring?)
- Your transformation strategy (what to keep, what to remove)
- Your performance requirements

## Installation

```bash
pip install textile-llm
```

**Note**: Package name is `textile-llm`, import name is `textile`.

```python
import textile  # Not textile-llm
```

## Quick Start

### 1. Copy a Reference Transformer

Browse [`examples/reference_transformers/`](examples/reference_transformers/) and find a pattern that matches your needs:

```bash
# Copy temporal decay transformer
cp examples/reference_transformers/temporal/decay.py my_app/transformers/
```

### 2. Use It

```python
import textile
from my_app.transformers.decay import DecayTransformer

# Configure transformers
textile.configure(
    transformers=[DecayTransformer(half_life_turns=5)]
)

# Use identical API to litellm.completion()
response = textile.completion(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

### 3. Customize for Your Domain

```python
# my_app/transformers/decay.py
class DecayTransformer(ContextTransformer):
    def transform(self, context, state):
        for msg in context.messages:
            age = state.turn_index - msg.turn_index

            # YOUR logic: Linear decay instead of exponential
            decay = max(0, 1 - (age / self.max_age))

            # YOUR rules: Keep important messages longer
            if msg.metadata._get_raw("important"):
                decay = max(decay, 0.8)

            msg.metadata.prominence *= decay

        # YOUR filtering strategy
        # ... remove based on YOUR criteria
        return context, state
```

That's it! Transformers are YOUR code.

## Why Textile?

### Drop-In Replacement

```python
# Before: litellm
import litellm
response = litellm.completion(model="gpt-4", messages=[...])

# After: textile (zero code changes!)
import textile
response = textile.completion(model="gpt-4", messages=[...])
```

All LiteLLM features work: streaming, async, tools, multimodal, 100+ models.

### Your Transformers, Your Rules

```python
class MyDomainTransformer(ContextTransformer):
    def transform(self, context, state):
        # YOUR business logic
        # YOUR decay formula
        # YOUR filtering rules
        # YOUR performance requirements
        return context, state
```

Not limited to our examples - build ANYTHING that transforms messages.

### Production Infrastructure

```python
# Testing utilities
from textile.utils.testing import TransformerTestCase

class TestMyTransformer(TransformerTestCase):
    def test_removes_old_messages(self):
        context, state = self.create_test_context([...])
        # ... test YOUR transformer

# Caching for embeddings
from textile.cache import MemoryCache
cache = MemoryCache()

# Observability
from textile.hooks import MetricsHook
hook = MetricsHook()
hook.register_callback(lambda m: print(f"Removed {m.messages_removed} messages"))
```

## Core Concepts

### Transformers

Implement the `ContextTransformer` protocol:

```python
from textile.transformers import ContextTransformer
from textile.core.context_window import ContextWindow
from textile.core.turn_state import TurnState

class MyTransformer(ContextTransformer):
    def transform(
        self,
        context: ContextWindow,
        state: TurnState
    ) -> tuple[ContextWindow, TurnState]:
        # Modify messages, state, or both
        return context, state

    def should_apply(
        self,
        context: ContextWindow,
        state: TurnState
    ) -> bool:
        # Optional: conditional execution
        return len(context.messages) > 5
```

### Transformation Pipeline

Chain transformers together:

```python
from my_app.transformers.decay import DecayTransformer
from my_app.transformers.pruning import SemanticPruningTransformer

textile.configure(
    transformers=[
        DecayTransformer(half_life_turns=5),    # First: age-based decay
        SemanticPruningTransformer(threshold=0.3),  # Then: relevance filtering
    ]
)
```

### Per-Call Override

```python
# Global config
textile.configure(transformers=[DecayTransformer()])

# Override for specific call
response = textile.completion(
    model="gpt-4",
    messages=[...],
    transformers=[MyCustomTransformer()]  # Replaces global
)
```

## Reference Transformers

**All transformers are in `examples/` - copy and customize them!**

| Pattern | Example | Use Case |
|---------|---------|----------|
| Temporal Filtering | `temporal/decay.py` | Age-based pruning |
| Semantic Filtering | `semantic/pruning.py` | Relevance filtering |
| Hybrid | `semantic/semantic_decay.py` | Time + relevance |
| State Transform | `semantic/tool_selection.py` | Tool/function filtering |

**Start here**: [`examples/quickstart/simple_filter.py`](examples/quickstart/simple_filter.py) - simplest possible transformer.

**Learn patterns**: [`examples/README.md`](examples/README.md) - comprehensive guide to building transformers.

## Features

### Full LiteLLM Compatibility

```python
# All litellm features work
response = textile.completion(
    model="gpt-4",
    messages=[...],
    temperature=0.7,
    max_tokens=100,
    tools=[...],
    response_format={"type": "json_object"},
    stream=True  # Streaming supported!
)

# Utility functions
model_info = textile.get_model_info("gpt-4")
supports_vision = textile.supports_vision("gpt-4o")
token_count = textile.count_tokens(messages=[...])
```

### Async/Await

```python
import asyncio

async def main():
    response = await textile.acompletion(
        model="gpt-4",
        messages=[...]
    )
    print(response.choices[0].message.content)

asyncio.run(main())
```

### Streaming

```python
response = textile.completion(
    model="gpt-4",
    messages=[...],
    stream=True
)

for chunk in response:
    if content := chunk.choices[0].delta.content:
        print(content, end="")
```

### Semantic Transformers

```python
from textile.embeddings import Embedding

textile.configure(
    embedding_model=Embedding("text-embedding-3-small"),
    transformers=[MySemanticTransformer()]
)
```

## Architecture

```
User Code
    ↓
textile.completion(messages=[...], transformers=[MyTransformer()])
    ↓
Transform messages → Apply YOUR transformers → Transform to LiteLLM format
    ↓
litellm.completion(transformed_messages)
    ↓
Return response (unchanged)
```

**Key Principles**:
- **Stateless**: No storage, pure transformation
- **Async-First**: All operations support async/await
- **Immutable**: Transformers return new objects
- **Zero API Changes**: Perfect LiteLLM drop-in

## Documentation

- **[Building Transformers Guide](examples/README.md)** - Comprehensive guide with patterns and best practices
- **[Reference Transformers](examples/reference_transformers/)** - Heavily-documented examples
- **[Migration Guide](MIGRATION.md)** - Upgrading from v0.3.x or v0.4.0 to v0.5.0
- **[API Documentation](#)** - Full API reference (coming soon)

## Examples

### Simple Filtering

```python
# examples/quickstart/simple_filter.py
class SimpleFilterTransformer(ContextTransformer):
    def transform(self, context, state):
        # Remove all user messages (for demonstration)
        user_ids = [m.id for m in context.messages if m.role == "user"]
        for msg_id in user_ids:
            context.remove_message(msg_id)
        return context, state
```

### Temporal Decay

```python
# examples/reference_transformers/temporal/decay.py
class DecayTransformer(ContextTransformer):
    def transform(self, context, state):
        current_turn = state.turn_index

        for msg in context.messages:
            age = current_turn - msg.turn_index
            decay_factor = 0.5 ** (age / self.half_life)
            msg.metadata.prominence *= decay_factor

        # Remove low-prominence messages
        # ... (see full implementation in examples/)
        return context, state
```

### Semantic Filtering

```python
# examples/reference_transformers/semantic/pruning.py
class SemanticPruningTransformer(ContextTransformer):
    def transform(self, context, state):
        query_embedding = state.user_embedding

        for msg in context.messages:
            if msg.embedding is not None:
                similarity = cosine_similarity(query_embedding, msg.embedding)
                if similarity < self.threshold:
                    context.remove_message(msg.id)

        return context, state
```

### Tool Selection

```python
# examples/reference_transformers/semantic/tool_selection.py
class SemanticToolSelectionTransformer(ContextTransformer):
    def transform(self, context, state):
        # Filter tools to most relevant
        selected_tools = self._select_relevant_tools(state.tools, state.user_embedding)

        # Return new state with filtered tools
        new_state = replace(state, tools=selected_tools)
        return context, new_state
```

## Development

```bash
# Clone repository
git clone https://github.com/blue-context/textile.git
cd textile

# Install with dev dependencies
uv sync --dev

# Run tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=textile --cov-report=html

# Lint
uv run ruff check textile/ tests/
uv run ruff format --check textile/ tests/
```

## Performance

Textile adds minimal overhead:
- **Transformation**: ~1-5ms (simple transformers)
- **Semantic operations**: ~10-50ms (with embedding cache)
- **Overhead**: Negligible vs. LLM API latency (100-1000ms+)

## Related Projects

- **[litellm](https://github.com/BerriAI/litellm)** - Python SDK for 100+ LLM providers (what Textile wraps)
- **[textile-go](https://github.com/blue-context/textile-go)** - Go implementation wrapping warp

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) (coming soon).

**Share your transformers**: If you build interesting transformers, consider sharing them! Community transformer examples are welcome.

## License

Copyright 2025 Blue Context Inc.

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

---

## FAQ

**Q: Where did DecayTransformer go?**

A: Removed in v0.5.0. Copy from `examples/reference_transformers/temporal/decay.py` to your project. Transformers are now YOUR code to customize for your domain.

**Q: Why not include production-ready transformers in the package?**

A: Transformers are highly domain-specific. What works for a chatbot doesn't work for RAG. What works for Q&A doesn't work for agents. Reference examples teach patterns - you customize for your domain.

**Q: Can I use the reference transformers in production?**

A: They're a starting point, not a production solution. Copy them, understand them, customize them for your requirements.

**Q: Do I have to write transformers from scratch?**

A: No! Start with a reference transformer that's close to your needs and modify it. That's what they're for.

**Q: What if I just want it to work out of the box?**

A: Textile v0.3.x included transformers, but they weren't tailored to anyone's specific needs. **v0.5.0 prioritizes giving you control** over providing generic solutions.

**Q: Can I share my transformers?**

A: Absolutely! We encourage it. Build a library of domain-specific transformers and share them.

**Q: Will Textile work without any transformers?**

A: Yes! It's a perfect LiteLLM drop-in even with `transformers=[]`. You can add transformers when you need them.

---

**Remember: Textile provides infrastructure. You provide the intelligence.**
