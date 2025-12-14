"""Embedding models for semantic search and similarity computation.

Provides embedding model abstractions for converting text to dense vectors.
Enables semantic similarity, context pruning, tool selection, and retrieval.

Example:
    >>> model = Embedding("text-embedding-3-small")
    >>> vector = model.encode("Hello world")
"""

from textile.embeddings.base import EmbeddingModel
from textile.embeddings.litellm import Embedding

__all__ = [
    "Embedding",
    "EmbeddingModel",
]
