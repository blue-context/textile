"""Async/sync compatibility helpers."""

import asyncio
from typing import TypeVar
from collections.abc import Coroutine

T = TypeVar('T')


def run_sync(coro: Coroutine[None, None, T]) -> T:
    """Run async coroutine in sync context with clear error if called from async.

    Raises:
        RuntimeError: If called from async context with guidance to use async variants
    """
    try:
        asyncio.get_running_loop()
        raise RuntimeError(
            "Cannot call sync API from async context. "
            "Use the async variant instead:\n"
            "  - Use acompletion() instead of completion()\n"
            "  - Use aembedding() instead of embedding()\n"
            "  - Or use 'await' if in async context"
        )
    except RuntimeError as e:
        # Distinguish between our error and "no running loop" from get_running_loop.
        if "no running event loop" in str(e).lower():
            return asyncio.run(coro)
        raise
