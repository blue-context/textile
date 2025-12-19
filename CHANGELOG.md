# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2024-12-18

### BREAKING CHANGES

**Transformer Implementations Removed**

Built-in transformer implementations have been completely removed from the core package. This represents an architectural shift from "middleware with transformers" to **"pure transformation infrastructure"**.

- ❌ **Removed**: `DecayTransformer`, `SemanticPruningTransformer`, `SemanticDecayTransformer`, `SemanticToolSelectionTransformer`
- ✅ **Available**: Reference implementations in `examples/reference_transformers/` directory
- ⚠️ **Migration Required**: Users must copy transformers from `examples/` to their projects and import from their own code

**Why This Change?**

Transformers are highly domain-specific. The reference implementations are teaching examples, not production solutions. You should build transformers tailored to your use case.

**Migration Path**

See [MIGRATION.md](MIGRATION.md) for the complete upgrade guide. Quick summary:

```python
# Before (v0.4.0 and earlier)
from textile.transformers import DecayTransformer

# After (v0.5.0)
# 1. Copy from examples/reference_transformers/temporal/decay.py to your project
# 2. Import from your own code
from my_app.transformers.decay import DecayTransformer
```

### Added

- **Testing Utilities** (`textile.utils.testing`)
  - `TransformerTestCase` base class for writing transformer tests
  - `create_context()` helper for test context creation
  - `assert_messages_removed()` and other testing assertions
  - Comprehensive test utilities for validating transformer behavior

- **Observability Hooks** (`textile.hooks`)
  - `MetricsHook` for performance tracking
  - Callback system for monitoring transformer execution
  - Duration tracking, message count metrics, and custom metadata

- **Reference Transformers** (`examples/reference_transformers/`)
  - Temporal transformers: `decay.py` (exponential time-based decay)
  - Semantic transformers: `pruning.py` (relevance-based filtering), `semantic_decay.py` (hybrid time + relevance), `tool_selection.py` (semantic tool filtering)
  - Fully documented reference implementations with customization guides
  - Production-ready patterns for building custom transformers

- **Building Transformers Guide** (`examples/README.md`)
  - Comprehensive guide with transformation patterns
  - Best practices and common pitfalls
  - Performance guidelines and optimization strategies
  - Step-by-step examples for creating custom transformers

- **Quickstart Examples** (`examples/quickstart/`)
  - Simple filter example demonstrating basic transformer structure
  - Minimal implementations for learning transformer concepts

### Changed

- **Package Positioning**: Textile is now positioned as pure infrastructure, not a transformer library
- **Transformer Exports**: `textile.transformers` now exports only infrastructure classes:
  - ✅ `ContextTransformer` (protocol/base class)
  - ✅ `TransformationPipeline` (orchestration)
  - ❌ No concrete transformer implementations (moved to examples)
- **Documentation**: Updated README.md and all documentation to reflect infrastructure-first positioning
- **PyPI Package Name**: Renamed to `textile-llm` for PyPI distribution

### Removed

- **Built-in Transformer Implementations**:
  - `textile/transformers/decay.py` → `examples/reference_transformers/temporal/decay.py`
  - `textile/transformers/semantic_prune.py` → `examples/reference_transformers/semantic/pruning.py`
  - `textile/transformers/semantic_decay.py` → `examples/reference_transformers/semantic/semantic_decay.py`
  - `textile/transformers/tool_selection.py` → `examples/reference_transformers/semantic/tool_selection.py`
- **Deprecation Warnings**: Removed all deprecation warnings from v0.4.0 (transformers no longer exist in core)

### Migration

Upgrading from v0.3.x or v0.4.0 requires copying transformers to your project. See [MIGRATION.md](MIGRATION.md) for:
- Step-by-step migration instructions
- Transformer mapping table (old import → new location)
- Complete migration examples
- Testing strategies post-migration

## [0.4.0] - 2024-12-18

### Deprecated

- **Built-in Transformers**: All built-in transformer implementations deprecated with warnings
  - `DecayTransformer`, `SemanticPruningTransformer`, `SemanticDecayTransformer`, `SemanticToolSelectionTransformer`
  - Added `DeprecationWarning` when importing from `textile.transformers`
  - Warnings directed users to copy from `examples/reference_transformers/`
  - Transformers continued to work in v0.4.0 (removed in v0.5.0)

### Added

- **Testing Infrastructure** (`textile.utils.testing`)
  - `TransformerTestCase` for writing transformer tests
  - Helper functions for creating test contexts
  - Assertion utilities for validating transformations

- **Observability System** (`textile.hooks`)
  - `MetricsHook` for tracking transformer performance
  - Callback system for monitoring execution
  - Support for custom metrics and monitoring

- **Reference Transformers** (`examples/reference_transformers/`)
  - Complete reference implementations in `examples/` directory
  - Heavily documented examples with customization guides
  - Production-ready patterns for building transformers

- **Transformer Building Guide** (`examples/README.md`)
  - Comprehensive guide for creating custom transformers
  - Transformation patterns and best practices
  - Performance guidelines and common pitfalls

- **Migration Documentation** (`MIGRATION.md`)
  - Step-by-step guide for migrating from v0.3.x to v0.4.0+
  - Transformer mapping table
  - Complete migration examples

### Changed

- **Package Description**: Updated to "transformation infrastructure" instead of "middleware with transformers"
- **Documentation**: All docs updated to reflect infrastructure-first positioning
- **Examples Structure**: Organized examples into logical categories (temporal, semantic, filtering, etc.)

## [0.3.0] - 2024-12-XX

### Added

- **Initial Release** with built-in transformers
  - `DecayTransformer`: Exponential time-based decay for message prominence
  - `SemanticPruningTransformer`: Relevance-based message filtering using embeddings
  - `SemanticDecayTransformer`: Hybrid transformer combining temporal and semantic factors
  - `SemanticToolSelectionTransformer`: Semantic filtering for tool/function selection

- **Core Infrastructure**
  - `ContextWindow`: Conversation history management
  - `Message`: Individual message representation with metadata
  - `TurnState`: Turn-level state and metadata
  - `TransformationPipeline`: Orchestration for multiple transformers

- **LiteLLM Wrapper**
  - `textile.completion()`: Drop-in replacement for `litellm.completion()`
  - `textile.acompletion()`: Async completion support
  - Automatic transformation pipeline integration
  - Zero API changes philosophy (fully compatible with LiteLLM)

- **Embedding Abstraction**
  - `Embedding` class for model-agnostic embeddings
  - Support for OpenAI, Anthropic, and custom embedding models
  - Caching support for improved performance

- **Configuration System**
  - `textile.configure()`: Global configuration
  - Support for custom transformers
  - Embedding model configuration
  - Pipeline customization

- **Testing Support**
  - pytest fixtures for testing transformers
  - Example test suites
  - CI/CD integration with GitHub Actions

### Documentation

- README.md with quickstart guide
- API documentation
- Transformer customization examples
- LiteLLM compatibility guide

## [0.2.0] - 2024-XX-XX

### Added

- Early development versions (internal releases)
- Prototype transformer implementations
- Basic LiteLLM integration

## [0.1.0] - 2024-XX-XX

### Added

- Initial project structure
- Core abstractions (ContextWindow, Message)
- Basic transformation pipeline

---

[unreleased]: https://github.com/blue-context/textile/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/blue-context/textile/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/blue-context/textile/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/blue-context/textile/releases/tag/v0.3.0
[0.2.0]: https://github.com/blue-context/textile/releases/tag/v0.2.0
[0.1.0]: https://github.com/blue-context/textile/releases/tag/v0.1.0
