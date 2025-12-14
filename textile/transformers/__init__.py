"""Context transformation pipeline and transformers.

Provides transformer infrastructure for modifying context windows and turn state.

Transformers:
- Temporal decay: Reduce prominence of older messages
- Semantic filtering: Remove off-topic messages
- Hybrid decay: Combine temporal and semantic relevance
- Tool selection: Filter large tool catalogs
- Pipeline orchestration: Chain multiple transformers

Example:
    >>> pipeline = TransformationPipeline([
    ...     DecayTransformer(half_life_turns=5),
    ...     SemanticPruningTransformer(threshold=0.3)
    ... ])
"""

from textile.transformers.base import ContextTransformer
from textile.transformers.decay import DecayTransformer
from textile.transformers.pipeline import TransformationPipeline
from textile.transformers.semantic_decay import SemanticDecayTransformer
from textile.transformers.semantic_prune import SemanticPruningTransformer
from textile.transformers.tool_selection import SemanticToolSelectionTransformer

__all__ = [
    "ContextTransformer",
    "DecayTransformer",
    "SemanticDecayTransformer",
    "SemanticPruningTransformer",
    "SemanticToolSelectionTransformer",
    "TransformationPipeline",
]
