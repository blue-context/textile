"""Internal utility functions for Textile.

- similarity: Cosine similarity for embedding vectors
- async_helpers: Safe async/sync interop utilities
"""

from textile.utils.async_helpers import run_sync
from textile.utils.similarity import cosine_similarity

__all__ = [
    "cosine_similarity",
    "run_sync",
]
