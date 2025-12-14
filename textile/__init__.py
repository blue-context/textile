"""Textile: Transparent context optimization middleware for LLMs.

Wraps LiteLLM with optional context optimization via transformation pipelines.

Example:
    >>> import textile
    >>> from textile.embeddings import Embedding
    >>> from textile.transformers import DecayTransformer
    >>>
    >>> textile.configure(
    ...     embedding_model=Embedding("text-embedding-3-small"),
    ...     transformers=[DecayTransformer()]
    ... )
    >>>
    >>> response = textile.completion(
    ...     model="gpt-4",
    ...     messages=[{"role": "user", "content": "Hello"}]
    ... )
"""

from textile.config import configure
from textile.lite import (
    acompletion,
    aembedding,
    aimage_generation,
    atranscription,
    batch_completion,
    batch_completion_models,
    batch_completion_models_all_responses,
    completion,
    embedding,
    get_model_info,
    image_generation,
    moderation,
    supports_function_calling,
    supports_response_schema,
    supports_vision,
    transcription,
)
from textile.lite.tokens import (
    count_tokens,
    create_pretrained_tokenizer,
    create_tokenizer,
    decode,
    encode,
)

__version__ = "0.3.0"

__all__ = [
    # Core completion APIs with context optimization
    "completion",
    "acompletion",
    # Embedding APIs with optional storage
    "embedding",
    "aembedding",
    # Batch operations
    "batch_completion",
    "batch_completion_models",
    "batch_completion_models_all_responses",
    # Multimodal
    "image_generation",
    "aimage_generation",
    "transcription",
    "atranscription",
    # Content safety
    "moderation",
    # Utilities
    "get_model_info",
    "supports_function_calling",
    "supports_vision",
    "supports_response_schema",
    # Configuration
    "configure",
    # Token counting - drop-in replacement for litellm tokenizer imports
    "count_tokens",
    "create_pretrained_tokenizer",
    "create_tokenizer",
    "encode",
    "decode",
]
