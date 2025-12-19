"""Metrics and observability hooks for transformers."""

import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class TransformerMetrics:
    """Metrics collected for a transformer execution.

    Attributes:
        transformer_name: Name of the transformer
        duration_ms: Execution time in milliseconds
        messages_before: Message count before transformation
        messages_after: Message count after transformation
        messages_removed: Number of messages removed
        should_apply: Whether should_apply returned True
        skipped: Whether transformer was skipped
        metadata: Custom metrics from the transformer
    """

    transformer_name: str
    duration_ms: float
    messages_before: int
    messages_after: int
    messages_removed: int
    should_apply: bool
    skipped: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def removal_rate(self) -> float:
        """Calculate percentage of messages removed.

        Returns:
            Removal rate as percentage (0.0-100.0)
        """
        if self.messages_before == 0:
            return 0.0
        return (self.messages_removed / self.messages_before) * 100.0


class MetricsHook:
    """Hook for collecting transformer metrics.

    Example:
        >>> hook = MetricsHook()
        >>> hook.on_transform_start("DecayTransformer", context, state)
        >>> # ... transformation happens ...
        >>> hook.on_transform_end("DecayTransformer", context, state)
        >>> metrics = hook.get_metrics()
        >>> print(f"Avg duration: {hook.avg_duration_ms():.2f}ms")
    """

    def __init__(self) -> None:
        """Initialize metrics hook."""
        self._metrics: list[TransformerMetrics] = []
        self._start_times: dict[str, float] = {}
        self._before_counts: dict[str, int] = {}
        self._callbacks: list[Callable[[TransformerMetrics], None]] = []

    def on_transform_start(
        self,
        transformer_name: str,
        messages_count: int,
        should_apply: bool,
    ) -> None:
        """Record transformer start.

        Args:
            transformer_name: Name of the transformer
            messages_count: Number of messages before transformation
            should_apply: Whether should_apply returned True
        """
        self._start_times[transformer_name] = time.time()
        self._before_counts[transformer_name] = messages_count

        # If should_apply is False, record as skipped
        if not should_apply:
            self._record_skip(transformer_name, messages_count)

    def on_transform_end(
        self,
        transformer_name: str,
        messages_count: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record transformer end.

        Args:
            transformer_name: Name of the transformer
            messages_count: Number of messages after transformation
            metadata: Optional custom metrics from transformer
        """
        if transformer_name not in self._start_times:
            return

        start_time = self._start_times.pop(transformer_name)
        duration_ms = (time.time() - start_time) * 1000

        before = self._before_counts.pop(transformer_name, messages_count)
        removed = before - messages_count

        metrics = TransformerMetrics(
            transformer_name=transformer_name,
            duration_ms=duration_ms,
            messages_before=before,
            messages_after=messages_count,
            messages_removed=removed,
            should_apply=True,
            skipped=False,
            metadata=metadata or {},
        )

        self._metrics.append(metrics)

        # Trigger callbacks
        for callback in self._callbacks:
            callback(metrics)

    def _record_skip(self, transformer_name: str, messages_count: int) -> None:
        """Record a skipped transformer.

        Args:
            transformer_name: Name of the transformer
            messages_count: Number of messages (unchanged)
        """
        metrics = TransformerMetrics(
            transformer_name=transformer_name,
            duration_ms=0.0,
            messages_before=messages_count,
            messages_after=messages_count,
            messages_removed=0,
            should_apply=False,
            skipped=True,
        )

        self._metrics.append(metrics)

        for callback in self._callbacks:
            callback(metrics)

    def register_callback(
        self,
        callback: Callable[[TransformerMetrics], None],
    ) -> None:
        """Register a callback for each transformer execution.

        Callbacks are called after each transformer finishes.

        Args:
            callback: Function taking TransformerMetrics

        Example:
            >>> def log_metrics(m: TransformerMetrics):
            ...     print(f"{m.transformer_name}: {m.duration_ms:.2f}ms")
            >>> hook.register_callback(log_metrics)
        """
        self._callbacks.append(callback)

    def get_metrics(self) -> list[TransformerMetrics]:
        """Get all collected metrics.

        Returns:
            List of TransformerMetrics
        """
        return self._metrics.copy()

    def get_metrics_by_transformer(
        self,
        transformer_name: str,
    ) -> list[TransformerMetrics]:
        """Get metrics for a specific transformer.

        Args:
            transformer_name: Name of transformer

        Returns:
            List of metrics for that transformer
        """
        return [m for m in self._metrics if m.transformer_name == transformer_name]

    def avg_duration_ms(self, transformer_name: str | None = None) -> float:
        """Calculate average execution duration.

        Args:
            transformer_name: Optional transformer to filter by

        Returns:
            Average duration in milliseconds
        """
        if transformer_name:
            metrics = self.get_metrics_by_transformer(transformer_name)
        else:
            metrics = [m for m in self._metrics if not m.skipped]

        if not metrics:
            return 0.0

        return sum(m.duration_ms for m in metrics) / len(metrics)

    def total_messages_removed(self, transformer_name: str | None = None) -> int:
        """Calculate total messages removed.

        Args:
            transformer_name: Optional transformer to filter by

        Returns:
            Total messages removed
        """
        if transformer_name:
            metrics = self.get_metrics_by_transformer(transformer_name)
        else:
            metrics = self._metrics

        return sum(m.messages_removed for m in metrics)

    def clear(self) -> None:
        """Clear all collected metrics."""
        self._metrics.clear()
        self._start_times.clear()
        self._before_counts.clear()

    def summary(self) -> dict[str, Any]:
        """Get summary statistics.

        Returns:
            Dict with summary stats
        """
        total_executions = len(self._metrics)
        skipped = len([m for m in self._metrics if m.skipped])
        executed = total_executions - skipped

        return {
            "total_executions": total_executions,
            "executed": executed,
            "skipped": skipped,
            "total_messages_removed": self.total_messages_removed(),
            "avg_duration_ms": self.avg_duration_ms(),
            "transformers": list(set(m.transformer_name for m in self._metrics)),
        }
