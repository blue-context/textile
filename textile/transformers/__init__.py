"""Transformation infrastructure for Textile.

Provides the base transformer protocol and pipeline orchestration.
Transformer implementations are in examples/reference_transformers/ -
copy them to your project and customize for your use case.
"""

from textile.transformers.base import ContextTransformer
from textile.transformers.pipeline import TransformationPipeline

__all__ = [
    "ContextTransformer",
    "TransformationPipeline",
]
