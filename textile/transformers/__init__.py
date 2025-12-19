"""Context transformation pipeline and transformers.

DEPRECATION NOTICE (v0.4.0):
============================
The built-in transformers (DecayTransformer, SemanticPruningTransformer, etc.)
are moving to examples/ in version 0.5.0.

These transformers are REFERENCE IMPLEMENTATIONS - teaching examples,
not production solutions. You should copy them to your project and
customize for your specific use case.

See: https://github.com/blue-context/textile/tree/main/examples/reference_transformers

Migration path:
1. Copy the transformer from examples/reference_transformers/
2. Place in your project (e.g., my_app/transformers/)
3. Import from your code: `from my_app.transformers.decay import DecayTransformer`

These transformers will be removed in v0.5.0.
"""

import warnings

from textile.transformers.base import ContextTransformer
from textile.transformers.pipeline import TransformationPipeline

# Deprecation message for transformers
_DEPRECATION_MESSAGE = (
    "Built-in transformers are deprecated and will be removed in v0.5.0. "
    "Copy from examples/reference_transformers/ to your project instead. "
    "See: https://github.com/blue-context/textile/tree/main/examples/reference_transformers"
)


def __getattr__(name: str):
    """Lazy import with deprecation warning for transformers."""
    if name == "DecayTransformer":
        warnings.warn(
            f"DecayTransformer is deprecated. {_DEPRECATION_MESSAGE}",
            DeprecationWarning,
            stacklevel=2,
        )
        from textile.transformers.decay import DecayTransformer

        return DecayTransformer

    if name == "SemanticPruningTransformer":
        warnings.warn(
            f"SemanticPruningTransformer is deprecated. {_DEPRECATION_MESSAGE}",
            DeprecationWarning,
            stacklevel=2,
        )
        from textile.transformers.semantic_prune import SemanticPruningTransformer

        return SemanticPruningTransformer

    if name == "SemanticDecayTransformer":
        warnings.warn(
            f"SemanticDecayTransformer is deprecated. {_DEPRECATION_MESSAGE}",
            DeprecationWarning,
            stacklevel=2,
        )
        from textile.transformers.semantic_decay import SemanticDecayTransformer

        return SemanticDecayTransformer

    if name == "SemanticToolSelectionTransformer":
        warnings.warn(
            f"SemanticToolSelectionTransformer is deprecated. {_DEPRECATION_MESSAGE}",
            DeprecationWarning,
            stacklevel=2,
        )
        from textile.transformers.tool_selection import SemanticToolSelectionTransformer

        return SemanticToolSelectionTransformer

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "ContextTransformer",
    "TransformationPipeline",
    # Deprecated (still exported for backwards compat until v0.5.0)
    "DecayTransformer",
    "SemanticPruningTransformer",
    "SemanticDecayTransformer",
    "SemanticToolSelectionTransformer",
]
