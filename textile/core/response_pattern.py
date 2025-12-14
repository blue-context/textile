"""Declarative pattern matching for response transformations."""

import inspect
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OnPattern:
    r"""Declarative pattern replacement for streaming LLM responses.

    Textile handles streaming complexity (boundaries, buffering, errors).

    Examples:
        >>> OnPattern('<PHONE_1>', '206-555-5555')
        >>> OnPattern(r'<PHONE_(\d+)>', lambda m: pii_map[f'<PHONE_{m.group(1)}>'])
        >>> OnPattern('<USER>', lambda: get_current_user())
    """

    pattern: str | re.Pattern
    replacement: str | Callable[[re.Match], str] | Callable[[], str]
    ignore_case: bool = False
    max_replacements: int = -1

    def __post_init__(self) -> None:
        """Validate and compile pattern."""
        if isinstance(self.pattern, str):
            flags = re.IGNORECASE if self.ignore_case else 0
            self.pattern = re.compile(re.escape(self.pattern), flags)
        elif isinstance(self.pattern, re.Pattern):
            if self.ignore_case and not (self.pattern.flags & re.IGNORECASE):
                logger.warning("ignore_case=True but pattern already compiled without IGNORECASE")
        else:
            raise TypeError(f"pattern must be str or re.Pattern, got {type(self.pattern).__name__}")

        assert isinstance(self.pattern, re.Pattern)

        if not (isinstance(self.replacement, str) or callable(self.replacement)):
            raise TypeError(
                f"replacement must be str or Callable, got {type(self.replacement).__name__}"
            )

    def get_replacement(self, match: re.Match) -> str:
        """Get replacement string for match."""
        if isinstance(self.replacement, str):
            return self.replacement

        if callable(self.replacement):
            sig = inspect.signature(self.replacement)
            if len(sig.parameters) == 0:
                return str(self.replacement())  # type: ignore[call-arg]
            return str(self.replacement(match))  # type: ignore[call-arg]

        raise TypeError(f"Unexpected replacement type: {type(self.replacement).__name__}")
