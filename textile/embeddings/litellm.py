"""LiteLLM embedding model implementation."""

import litellm
import numpy as np
import numpy.typing as npt

from textile.embeddings.base import EmbeddingModel


class Embedding(EmbeddingModel):
    """LiteLLM embedding model.

    Supports OpenAI, Cohere, Voyage AI, and more via LiteLLM.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
        **litellm_kwargs,
    ) -> None:
        """Initialize embedding model."""
        self.model = model
        self._dimensions = dimensions
        self.litellm_kwargs = litellm_kwargs

        # Auto-detect dimension if not provided
        if self._dimensions is None:
            test_response = litellm.embedding(model=self.model, input="test")
            self._dimensions = len(test_response.data[0]["embedding"])

    def encode(self, text: str) -> npt.NDArray[np.float32]:
        """Encode text to embedding vector."""
        response = litellm.embedding(model=self.model, input=text, **self.litellm_kwargs)
        return np.array(response.data[0]["embedding"], dtype=np.float32)

    def encode_batch(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Encode multiple texts to embedding vectors."""
        response = litellm.embedding(model=self.model, input=texts, **self.litellm_kwargs)
        embeddings = [data["embedding"] for data in response.data]
        return np.array(embeddings, dtype=np.float32)

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimensions
