"""Configuration management for Textile."""

from textile.embeddings.base import EmbeddingModel
from textile.transformers.base import ContextTransformer


class TextileConfig:
    """Global configuration for embedding model and transformers.

    Warning:
        Mutable global state shared by all completion() calls.
        Not thread-safe - configure once at startup.
    """

    def __init__(self) -> None:
        """Initialize empty configuration."""
        self._embedding_model: EmbeddingModel | None = None
        self._transformers: list[ContextTransformer] = []

    @property
    def embedding_model(self) -> EmbeddingModel | None:
        """Embedding model for semantic operations."""
        return self._embedding_model

    @embedding_model.setter
    def embedding_model(self, value: EmbeddingModel) -> None:
        """Set embedding model."""
        self._embedding_model = value

    @property
    def transformers(self) -> list[ContextTransformer]:
        """Global transformation pipeline."""
        return self._transformers

    @transformers.setter
    def transformers(self, value: list[ContextTransformer]) -> None:
        """Set transformation pipeline."""
        self._transformers = value


_config = TextileConfig()


def get_config() -> TextileConfig:
    """Get global configuration instance."""
    return _config


def configure(
    embedding_model: EmbeddingModel | None = None,
    transformers: list[ContextTransformer] | None = None,
) -> None:
    """Configure Textile globally.

    Warning:
        Mutable global state. Configure once at startup.
        Not thread-safe. In tests, mock textile.config.get_config().

    Args:
        embedding_model: Embedding model for semantic transformers.
        transformers: Pipeline applied to all completions.
            Per-call transformers override (replace, not extend).

    Example:
        >>> import textile
        >>> from textile.embeddings import Embedding
        >>> from textile.transformers import DecayTransformer
        >>> textile.configure(
        ...     embedding_model=Embedding("text-embedding-3-small"),
        ...     transformers=[DecayTransformer(half_life_turns=5)]
        ... )
    """
    config = get_config()

    if embedding_model is not None:
        config.embedding_model = embedding_model

    if transformers is not None:
        config.transformers = transformers
