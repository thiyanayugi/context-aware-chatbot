"""
Input filtering and sanitization.

Design Decisions:
- Defense in depth: Multiple detection layers
- Configurable: Patterns and sensitivity adjustable
- Non-blocking: Returns sanitized input, doesn't raise
- Logging: All detections are logged for monitoring
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional


logger = logging.getLogger(__name__)


# Common prompt injection patterns
INJECTION_PATTERNS = [
    # Direct instruction override attempts
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
    r"forget\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",

    # System prompt extraction attempts
    r"(what|show|tell|reveal|display)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions?|rules?)",
    r"repeat\s+(your\s+)?(system\s+)?(prompt|instructions?)",

    # Role-playing attacks
    r"you\s+are\s+now\s+(a|an|the)\s+",
    r"pretend\s+(to\s+be|you\s+are)\s+",
    r"act\s+as\s+(if\s+you\s+are\s+)?(a|an|the)\s+",

    # Jailbreak attempts
    r"(DAN|STAN|DUDE)\s+mode",
    r"developer\s+mode",
    r"(enable|activate)\s+(jailbreak|unrestricted)\s+mode",

    # Delimiter injection
    r"<\|?(system|assistant|user)\|?>",
    r"\[INST\]|\[\/INST\]",
    r"###\s*(system|instruction|human|assistant)",
]

# Compiled patterns for efficiency
COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


@dataclass
class FilterResult:
    """Result of input filtering."""
    original: str
    filtered: str
    is_suspicious: bool
    detections: list[str]
    risk_score: float  # 0.0 to 1.0


class InputFilter:
    """
    Filters and sanitizes user input.

    Detects prompt injection attempts and other malicious patterns.
    """

    def __init__(
        self,
        patterns: Optional[list[re.Pattern]] = None,
        max_length: int = 10000,
        strip_control_chars: bool = True
    ):
        """
        Initialize input filter.

        Args:
            patterns: Custom regex patterns (uses defaults if None)
            max_length: Maximum allowed input length
            strip_control_chars: Whether to remove control characters
        """
        self.patterns = patterns or COMPILED_PATTERNS
        self.max_length = max_length
        self.strip_control_chars = strip_control_chars

    def filter(self, text: str) -> FilterResult:
        """
        Filter and sanitize input text.

        Args:
            text: Raw user input

        Returns:
            FilterResult with sanitized text and detection info
        """
        detections = []
        filtered = text

        # Length check
        if len(text) > self.max_length:
            filtered = text[:self.max_length]
            detections.append(f"truncated_length:{len(text)}")
            logger.warning(f"Input truncated from {len(text)} to {self.max_length} chars")

        # Control character removal
        if self.strip_control_chars:
            original_len = len(filtered)
            filtered = self._strip_control_chars(filtered)
            if len(filtered) < original_len:
                detections.append("control_chars_removed")

        # Pattern matching
        for i, pattern in enumerate(self.patterns):
            matches = pattern.findall(filtered)
            if matches:
                detection = f"pattern_{i}:{matches[0] if isinstance(matches[0], str) else matches[0][0]}"
                detections.append(detection)
                logger.warning(f"Injection pattern detected: {detection}")

        # Calculate risk score
        risk_score = min(1.0, len(detections) * 0.3)

        return FilterResult(
            original=text,
            filtered=filtered,
            is_suspicious=len(detections) > 0,
            detections=detections,
            risk_score=risk_score
        )

    def is_safe(self, text: str, threshold: float = 0.5) -> bool:
        """
        Quick check if input is safe.

        Args:
            text: Input to check
            threshold: Risk score threshold

        Returns:
            True if input is considered safe
        """
        result = self.filter(text)
        return result.risk_score < threshold

    def sanitize(self, text: str) -> str:
        """
        Sanitize input and return cleaned text.

        Args:
            text: Raw input

        Returns:
            Sanitized text
        """
        return self.filter(text).filtered

    def _strip_control_chars(self, text: str) -> str:
        """Remove control characters except common whitespace."""
        # Keep: tab, newline, carriage return, space
        # Remove: other control characters
        return "".join(
            char for char in text
            if char >= " " or char in "\t\n\r"
        )


def detect_prompt_injection(text: str) -> tuple[bool, list[str]]:
    """
    Convenience function to detect prompt injection.

    Args:
        text: Input to check

    Returns:
        Tuple of (is_suspicious, list of detections)
    """
    filter_instance = InputFilter()
    result = filter_instance.filter(text)
    return result.is_suspicious, result.detections
