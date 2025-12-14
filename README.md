# Textile

Transparent message transformation middleware for LLM applications in Python.

Textile wraps [LiteLLM](https://github.com/BerriAI/litellm) with a powerful transformation pipeline system while maintaining zero API changes for consumers. This allows you to apply configurable transformations to messages before LLM calls transparently.

## Features

- **Zero API Changes**: Drop-in replacement for `litellm.completion()`
- **Transformation Pipeline**: Apply multiple transformers in sequence
- **Built-in Transformers**: Temporal decay, semantic pruning, tool selection
- **Streaming Support**: Transform streaming chunks in real-time
- **Async-First**: Native async/await support with sync wrappers
- **Stateless**: No storage required, pure transformation middleware
- **Composable**: Build complex pipelines with simple transformers

## Installation

```bash
pip install textile-llm
```

**Note**: The package is named `textile-llm` on PyPI, but you import it as `textile`:

```python
import textile  # Import name stays the same
```

## Quick Start

```python
import textile
from textile.transformers import DecayTransformer

# Configure transformation pipeline
textile.configure(
    transformers=[
        DecayTransformer(half_life_turns=5)
    ]
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

That's it! Textile automatically applies transformers to optimize your context window.

## Core Concepts

### Transformers

Transformers modify the context window before sending to the LLM. Each transformer implements:

```python
from textile.transformers import ContextTransformer
from textile.core.context_window import ContextWindow
from textile.core.turn_state import TurnState

class MyTransformer(ContextTransformer):
    def transform(
        self,
        context: ContextWindow,
        state: TurnState
    ) -> ContextWindow:
        # Modify context.messages
        # Return new ContextWindow
        return context
```

### Built-in Transformers

#### DecayTransformer

Reduces prominence of older messages with exponential decay:

```python
from textile.transformers import DecayTransformer

textile.configure(
    transformers=[
        DecayTransformer(
            half_life_turns=5,  # Messages lose 50% weight every 5 turns
            min_weight=0.1      # Minimum weight threshold
        )
    ]
)
```

#### SemanticPruningTransformer

Removes messages below a similarity threshold:

```python
from textile.transformers import SemanticPruningTransformer
from textile.embeddings import Embedding

textile.configure(
    embedding_model=Embedding("text-embedding-3-small"),
    transformers=[
        SemanticPruningTransformer(
            similarity_threshold=0.3  # Remove messages with < 0.3 similarity
        )
    ]
)
```

#### SemanticDecayTransformer

Combines temporal and semantic decay:

```python
from textile.transformers import SemanticDecayTransformer
from textile.embeddings import Embedding

textile.configure(
    embedding_model=Embedding("text-embedding-3-small"),
    transformers=[
        SemanticDecayTransformer(
            half_life_turns=5,
            similarity_weight=0.7,  # 70% semantic, 30% temporal
            min_weight=0.1
        )
    ]
)
```

#### SemanticToolSelectionTransformer

Filters large tool catalogs to most relevant tools:

```python
from textile.transformers import SemanticToolSelectionTransformer
from textile.embeddings import Embedding

textile.configure(
    embedding_model=Embedding("text-embedding-3-small"),
    transformers=[
        SemanticToolSelectionTransformer(
            top_k=5,  # Keep top 5 most relevant tools
            similarity_threshold=0.3
        )
    ]
)
```

### Transformation Pipeline

Transformers execute in sequence:

```python
from textile.transformers import TransformationPipeline

pipeline = TransformationPipeline([
    DecayTransformer(half_life_turns=5),
    SemanticPruningTransformer(similarity_threshold=0.3),
])

textile.configure(transformers=pipeline.transformers)
```

### Per-Call Transformers

Override global transformers for specific calls:

```python
# Global configuration
textile.configure(transformers=[DecayTransformer()])

# Override for this call only
response = textile.completion(
    model="gpt-4",
    messages=[...],
    transformers=[SemanticPruningTransformer()]  # Replaces global config
)
```

## Streaming

Textile supports streaming with transparent transformation:

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

## Async Support

Native async/await support:

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

## LiteLLM Compatibility

Textile is a drop-in replacement for LiteLLM. All LiteLLM features work:

```python
import textile

# All litellm.completion() parameters supported
response = textile.completion(
    model="gpt-4",
    messages=[...],
    temperature=0.7,
    max_tokens=100,
    tools=[...],
    response_format={"type": "json_object"}
)

# Utility functions
info = textile.get_model_info("gpt-4")
supports_vision = textile.supports_vision("gpt-4o")
token_count = textile.count_tokens(messages=[...])

# Batch operations
responses = textile.batch_completion(
    models=["gpt-4", "claude-3-opus"],
    messages=[...]
)

# Multimodal
image = textile.image_generation(
    model="dall-e-3",
    prompt="A sunset over mountains"
)
```

## Architecture

Textile uses an async-first, immutable architecture:

1. **Request Phase**:
   - Convert messages to `ContextWindow`
   - Apply transformers sequentially
   - Send transformed messages to LiteLLM
   - Return response unchanged

2. **Streaming Phase**:
   - Apply transformers to context
   - Stream chunks through `StreamingResponseHandler`
   - Transform chunks in real-time if needed

### Design Principles

- **Stateless**: No storage, no persistence, pure transformation
- **Async-First**: All operations support async/await
- **Immutable**: Transformers return new objects, never mutate
- **Zero API Changes**: Perfect drop-in replacement for LiteLLM

## Configuration

Configure Textile globally at startup:

```python
from textile import configure
from textile.embeddings import Embedding
from textile.transformers import DecayTransformer

configure(
    embedding_model=Embedding("text-embedding-3-small"),
    transformers=[DecayTransformer(half_life_turns=5)]
)
```

**Warning**: Global configuration is not thread-safe. Configure once at startup.

## Environment Variables

Textile respects LiteLLM environment variables:

```bash
# API Keys (LiteLLM standard)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Embedding configuration (for semantic transformers)
export EMBEDDING_MODEL="text-embedding-3-small"
export OPENAI_API_KEY="sk-..."  # If using OpenAI embeddings
```

## Development

```bash
# Clone the repository
git clone https://github.com/blue-context/textile.git
cd textile

# Install with dev dependencies using uv
uv sync --dev

# Run tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=textile --cov-report=html

# Run linting
uv run ruff check textile/ tests/
uv run ruff format --check textile/ tests/
```

## Performance

Textile adds minimal overhead:
- Transformation: ~1-5ms depending on pipeline complexity
- Semantic operations: ~10-50ms for embedding calls
- Overhead is negligible compared to LLM API latency (100-1000ms+)

## Related Projects

- [litellm](https://github.com/BerriAI/litellm) - Python SDK for 100+ LLM providers
- [textile-go](https://github.com/blue-context/textile-go) - Go implementation wrapping warp

## License

Copyright 2025 Blue Context Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
