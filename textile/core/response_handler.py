"""Streaming response handler for pattern-based transformations."""

import logging
import re

from textile.core.response_pattern import OnPattern

logger = logging.getLogger(__name__)

MAX_BUFFER_SIZE = 10240
DEFAULT_SAFETY_MARGIN = 50
MIN_BUFFER_SIZE = 50


class StreamingResponseHandler:
    """Handle pattern-based transformation of streaming LLM responses.

    Buffers chunks to avoid splitting patterns, applies replacements, handles errors.
    Processes both streaming and non-streaming uniformly.
    """

    def __init__(self, patterns: list[OnPattern], max_buffer_size: int = MAX_BUFFER_SIZE):
        """Initialize response handler."""
        self.patterns = patterns
        self.max_buffer_size = max_buffer_size
        self.buffer = ""
        self.buffer_threshold = self._calculate_buffer_threshold()
        self.stats = {
            'chunks_processed': 0,
            'patterns_applied': 0,
            'errors': 0,
        }

    def _calculate_buffer_threshold(self) -> int:
        """Calculate buffer retention size from pattern lengths."""
        if not self.patterns:
            return MIN_BUFFER_SIZE

        max_pattern_len = MIN_BUFFER_SIZE
        for pattern in self.patterns:
            assert isinstance(pattern.pattern, re.Pattern)
            pattern_str = pattern.pattern.pattern
            max_pattern_len = max(max_pattern_len, len(pattern_str))

        return min(max_pattern_len + DEFAULT_SAFETY_MARGIN, self.max_buffer_size // 2)

    def transform_chunk(self, chunk: str) -> str:
        """Transform chunk, buffering incomplete patterns."""
        if not chunk:
            return ""

        self.stats['chunks_processed'] += 1
        self.buffer += chunk
        safe_boundary = self._find_safe_boundary()

        if safe_boundary <= 0:
            return ""

        to_process = self.buffer[:safe_boundary]
        self.buffer = self.buffer[safe_boundary:]

        try:
            return self._apply_patterns(to_process)
        except Exception as e:
            logger.error(f"Error transforming chunk: {e}", exc_info=True)
            self.stats['errors'] += 1
            return to_process

    def _find_safe_boundary(self) -> int:
        """Find index for safe processing without splitting patterns."""
        buffer_len = len(self.buffer)

        if buffer_len < self.buffer_threshold and buffer_len < self.max_buffer_size:
            return 0

        safe_boundary = buffer_len - self.buffer_threshold

        if buffer_len > self.max_buffer_size:
            logger.warning(f"Buffer exceeded max size ({self.max_buffer_size}), forcing flush")
            return max(0, safe_boundary)

        for pattern in self.patterns:
            assert isinstance(pattern.pattern, re.Pattern)
            for match in pattern.pattern.finditer(self.buffer):
                if match.start() < safe_boundary < match.end():
                    safe_boundary = match.start()

            safe_boundary = self._adjust_for_partial_pattern(safe_boundary, pattern.pattern)

        return max(0, safe_boundary)

    def _adjust_for_partial_pattern(self, boundary: int, pattern: re.Pattern) -> int:
        """Adjust boundary to avoid splitting starting pattern."""
        if boundary <= 0 or boundary >= len(self.buffer):
            return boundary

        search_start = max(0, boundary - self.buffer_threshold)
        search_region = self.buffer[search_start:boundary + self.buffer_threshold]

        for i in range(len(search_region)):
            if (match := pattern.match(search_region[i:])):
                match_start_in_buffer = search_start + i
                match_end_in_buffer = search_start + i + match.end()

                if match_start_in_buffer < boundary < match_end_in_buffer:
                    return match_start_in_buffer

        return boundary

    def _apply_patterns(self, text: str) -> str:
        """Apply pattern handlers sequentially."""
        if not self.patterns:
            return text

        result = text

        for pattern_handler in self.patterns:
            try:
                result = self._apply_single_pattern(result, pattern_handler)
            except Exception as e:
                assert isinstance(pattern_handler.pattern, re.Pattern)
                logger.error(f"Error applying pattern {pattern_handler.pattern.pattern}: {e}", exc_info=True)
                self.stats['errors'] += 1

        return result

    def _apply_single_pattern(self, text: str, handler: OnPattern) -> str:
        """Apply single OnPattern handler."""
        replacements_made = 0

        def replace_func(match: re.Match) -> str:
            nonlocal replacements_made

            if handler.max_replacements >= 0 and replacements_made >= handler.max_replacements:
                return str(match.group(0))

            try:
                replacement = handler.get_replacement(match)
                replacements_made += 1
                self.stats['patterns_applied'] += 1
                return str(replacement)
            except Exception as e:
                logger.error(f"Error in replacement function: {e}", exc_info=True)
                self.stats['errors'] += 1
                return str(match.group(0))

        assert isinstance(handler.pattern, re.Pattern)
        return str(handler.pattern.sub(replace_func, text))

    def flush(self) -> str:
        """Flush remaining buffer at stream end."""
        if not self.buffer:
            return ""

        try:
            transformed = self._apply_patterns(self.buffer)
            self.buffer = ""
            return transformed
        except Exception as e:
            logger.error(f"Error flushing buffer: {e}", exc_info=True)
            self.stats['errors'] += 1
            result = self.buffer
            self.buffer = ""
            return result

    def get_stats(self) -> dict:
        """Get processing statistics."""
        return self.stats.copy()
