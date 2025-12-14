"""Direct re-exports from litellm for APIs Textile doesn't wrap.

Textile wraps completion() and embedding() to add conversation context
optimization. All other litellm APIs are re-exported directly to maintain
perfect 1:1 compatibility with zero maintenance overhead.

For documentation on these functions, see: https://docs.litellm.ai/docs/
"""

from typing import Any

import litellm
from litellm import (
    # Batch operations
    batch_completion,
    batch_completion_models,
    batch_completion_models_all_responses,
    # Multimodal
    image_generation,
    aimage_generation,
    transcription,
    atranscription,
    # Model utilities
    get_model_info,
    supports_function_calling,
    supports_vision,
    supports_response_schema,
)


def moderation(
    input: str | list[str],
    model: str = "text-moderation-stable",
    **litellm_kwargs: Any,
) -> Any:
    """Check content against moderation policies.

    Provides default model for backward compatibility since litellm.moderation
    has model=None as default.
    """
    return litellm.moderation(
        input=input,
        model=model,
        **litellm_kwargs,
    )


__all__ = [
    # Batch operations
    "batch_completion",
    "batch_completion_models",
    "batch_completion_models_all_responses",
    # Multimodal
    "image_generation",
    "aimage_generation",
    "transcription",
    "atranscription",
    "moderation",
    # Model utilities
    "get_model_info",
    "supports_function_calling",
    "supports_vision",
    "supports_response_schema",
]
