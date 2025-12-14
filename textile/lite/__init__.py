"""LiteLLM compatibility layer.

This module provides a drop-in replacement for LiteLLM's API,
adding Textile's conversation context optimization features.
"""

from textile.lite.completion import acompletion, completion
from textile.lite.embeddings import aembedding, embedding
from textile.lite.exports import (
    # Batch operations
    batch_completion,
    batch_completion_models,
    batch_completion_models_all_responses,
    # Multimodal
    aimage_generation,
    atranscription,
    image_generation,
    moderation,
    transcription,
    # Utilities
    get_model_info,
    supports_function_calling,
    supports_response_schema,
    supports_vision,
)
from textile.lite.tokens import (
    count_tokens,
    create_pretrained_tokenizer,
    create_tokenizer,
    decode,
    encode,
)

__all__ = [
    # Core completion
    "completion",
    "acompletion",
    # Embeddings
    "embedding",
    "aembedding",
    # Batch
    "batch_completion",
    "batch_completion_models",
    "batch_completion_models_all_responses",
    # Multimodal
    "image_generation",
    "aimage_generation",
    "transcription",
    "atranscription",
    "moderation",
    # Utilities
    "get_model_info",
    "supports_function_calling",
    "supports_vision",
    "supports_response_schema",
    # Token utilities
    "count_tokens",
    "create_pretrained_tokenizer",
    "create_tokenizer",
    "encode",
    "decode",
]
