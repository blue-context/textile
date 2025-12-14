"""Abstract base class for embedding models."""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class EmbeddingModel(ABC):
    """Abstract interface for text embedding models.

    Implementations must provide encode(), encode_batch(), and dimension property.
    Vectors should be float32 and thread-safe.
    """

    @abstractmethod
    def encode(self, text: str) -> npt.NDArray[np.float32]:
        """Encode text to embedding vector."""
        pass

    @abstractmethod
    def encode_batch(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Encode multiple texts to embedding vectors."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get embedding dimension."""
        pass
