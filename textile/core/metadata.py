"""Message metadata with global properties and typed namespaces."""

from dataclasses import asdict, dataclass
from typing import Any, Protocol, TypeVar

T = TypeVar("T", bound="TransformerMetadata")


class TransformerMetadata(Protocol):
    """Protocol for transformer-specific metadata."""

    def validate(self) -> None:
        """Validate metadata values."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict."""
        ...

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Create from dict."""
        ...


@dataclass
class DataclassMetadata:
    """Mixin providing default TransformerMetadata implementations."""

    def validate(self) -> None:
        """Override to add validation."""
        pass

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Create from dict."""
        return cls(**data)


class MessageMetadata:
    """Two-tier metadata: global properties and typed namespaces.

    Global properties: prominence, turn_index, embedding
    Namespaces: transformer-specific typed metadata in isolated namespaces

    Example:
        >>> metadata = MessageMetadata()
        >>> metadata.prominence = 0.85
        >>> decay_meta = SemanticDecayMetadata(salience=0.8, age_hours=24)
        >>> metadata.set_namespace("semantic_decay", decay_meta)
        >>> decay = metadata.get_namespace("semantic_decay", SemanticDecayMetadata)
    """

    def __init__(self) -> None:
        """Initialize metadata with empty properties and namespaces."""
        self._global: dict[str, Any] = {}
        self._namespaces: dict[str, dict[str, Any]] = {}

    @property
    def prominence(self) -> float:
        """Relevance score (0.0-1.0)."""
        return float(self._global.get("prominence", 1.0))

    @prominence.setter
    def prominence(self, value: float) -> None:
        """Set relevance score, clamped to [0.0, 1.0]."""
        if value < 0.0:
            raise ValueError(f"prominence must be >= 0.0, got {value}")
        self._global["prominence"] = min(value, 1.0)

    @property
    def turn_index(self) -> int:
        """Turn when created."""
        return int(self._global.get("turn_index", 0))

    @turn_index.setter
    def turn_index(self, value: int) -> None:
        """Set turn index."""
        if value < 0:
            raise ValueError(f"turn_index must be >= 0, got {value}")
        self._global["turn_index"] = value

    @property
    def embedding(self) -> list[float] | None:
        """Semantic vector."""
        return self._global.get("embedding")

    @embedding.setter
    def embedding(self, value: list[float] | None) -> None:
        """Set semantic vector embedding."""
        self._global["embedding"] = value

    def get_namespace(
        self,
        name: str,
        metadata_type: type[T],
    ) -> T | None:
        """Retrieve typed metadata from namespace."""
        if (data := self._namespaces.get(name)) is None:
            return None
        return metadata_type.from_dict(data)

    def set_namespace(self, name: str, metadata: Any) -> None:
        """Store typed metadata with validation."""
        metadata.validate()
        self._namespaces[name] = metadata.to_dict()

    def has_namespace(self, name: str) -> bool:
        """Check if namespace exists."""
        return name in self._namespaces

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "global": self._global.copy(),
            "namespaces": {k: v.copy() for k, v in self._namespaces.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MessageMetadata":
        """Deserialize from dict."""
        metadata = cls()
        metadata._global = data.get("global", {}).copy()
        metadata._namespaces = {k: v.copy() for k, v in data.get("namespaces", {}).items()}
        return metadata

    def _get_raw(self, key: str) -> Any:
        """Get raw global value (backward compatibility)."""
        return self._global.get(key)

    def _set_raw(self, key: str, value: Any) -> None:
        """Set raw global value (backward compatibility)."""
        self._global[key] = value

    def _contains(self, key: str) -> bool:
        """Check if key exists in global (backward compatibility)."""
        return key in self._global
