"""
Output validation.

Design Decisions:
- Multi-layered: Multiple validation checks
- Configurable: Thresholds and rules adjustable
- Non-destructive: Returns validation result, caller decides action
- Extensible: Easy to add new validators
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum


logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """A single validation issue."""
    code: str
    message: str
    level: ValidationLevel


@dataclass
class ValidationResult:
    """Result of output validation."""
    is_valid: bool
    content: str
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Check if there are error-level issues."""
        return any(
            issue.level in (ValidationLevel.ERROR, ValidationLevel.CRITICAL)
            for issue in self.issues
        )

    @property
    def has_warnings(self) -> bool:
        """Check if there are warning-level issues."""
        return any(issue.level == ValidationLevel.WARNING for issue in self.issues)


class OutputValidator:
    """
    Validates LLM output for safety and quality.

    Checks for empty responses, length limits, and unsafe patterns.
    """

    def __init__(
        self,
        min_length: int = 1,
        max_length: int = 10000,
        block_unsafe_tools: bool = True,
        custom_validators: Optional[list[Callable[[str], Optional[ValidationIssue]]]] = None
    ):
        """
        Initialize output validator.

        Args:
            min_length: Minimum response length
            max_length: Maximum response length
            block_unsafe_tools: Block responses with tool execution patterns
            custom_validators: Additional validation functions
        """
        self.min_length = min_length
        self.max_length = max_length
        self.block_unsafe_tools = block_unsafe_tools
        self.custom_validators = custom_validators or []

    def validate(self, content: str) -> ValidationResult:
        """
        Validate LLM output.

        Args:
            content: LLM response content

        Returns:
            ValidationResult with issues found
        """
        issues = []

        # Empty check
        if not content or not content.strip():
            issues.append(ValidationIssue(
                code="EMPTY_RESPONSE",
                message="Response is empty",
                level=ValidationLevel.ERROR
            ))
            return ValidationResult(is_valid=False, content=content, issues=issues)

        stripped = content.strip()

        # Length checks
        if len(stripped) < self.min_length:
            issues.append(ValidationIssue(
                code="TOO_SHORT",
                message=f"Response too short: {len(stripped)} < {self.min_length}",
                level=ValidationLevel.WARNING
            ))

        if len(stripped) > self.max_length:
            issues.append(ValidationIssue(
                code="TOO_LONG",
                message=f"Response too long: {len(stripped)} > {self.max_length}",
                level=ValidationLevel.WARNING
            ))

        # Unsafe tool patterns
        if self.block_unsafe_tools:
            tool_issues = self._check_unsafe_tools(stripped)
            issues.extend(tool_issues)

        # Refusal detection (model refused to answer)
        refusal_issue = self._check_refusal(stripped)
        if refusal_issue:
            issues.append(refusal_issue)

        # Hallucination indicators
        hallucination_issue = self._check_hallucination_patterns(stripped)
        if hallucination_issue:
            issues.append(hallucination_issue)

        # Custom validators
        for validator in self.custom_validators:
            issue = validator(stripped)
            if issue:
                issues.append(issue)

        # Determine overall validity
        is_valid = not any(
            issue.level in (ValidationLevel.ERROR, ValidationLevel.CRITICAL)
            for issue in issues
        )

        return ValidationResult(
            is_valid=is_valid,
            content=content,
            issues=issues
        )

    def _check_unsafe_tools(self, content: str) -> list[ValidationIssue]:
        """Check for unsafe tool execution patterns."""
        issues = []

        # Patterns that might indicate tool execution attempts
        unsafe_patterns = [
            (r"```\s*(bash|shell|sh)\s*\n.*?(rm\s+-rf|sudo|chmod|chown)", "SHELL_COMMAND"),
            (r"<script[^>]*>.*?</script>", "SCRIPT_TAG"),
            (r"javascript:\s*", "JAVASCRIPT_URI"),
            (r"eval\s*\(", "EVAL_CALL"),
            (r"exec\s*\(", "EXEC_CALL"),
            (r"__import__\s*\(", "IMPORT_CALL"),
        ]

        for pattern, code in unsafe_patterns:
            if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
                issues.append(ValidationIssue(
                    code=f"UNSAFE_{code}",
                    message=f"Potentially unsafe pattern detected: {code}",
                    level=ValidationLevel.WARNING
                ))
                logger.warning(f"Unsafe pattern in output: {code}")

        return issues

    def _check_refusal(self, content: str) -> Optional[ValidationIssue]:
        """Check if the model refused to answer."""
        refusal_phrases = [
            "I cannot",
            "I'm not able to",
            "I am not able to",
            "I won't be able to",
            "I'm sorry, but I cannot",
            "As an AI language model, I cannot",
            "I don't have the ability to",
        ]

        content_lower = content.lower()
        for phrase in refusal_phrases:
            if phrase.lower() in content_lower:
                return ValidationIssue(
                    code="MODEL_REFUSAL",
                    message="Model may have refused the request",
                    level=ValidationLevel.INFO
                )

        return None

    def _check_hallucination_patterns(self, content: str) -> Optional[ValidationIssue]:
        """Check for common hallucination indicators."""
        # Patterns that might indicate hallucination
        patterns = [
            r"As of my knowledge cutoff",
            r"As of my last update",
            r"I don't have access to real-time",
            r"I cannot browse the internet",
        ]

        for pattern in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return ValidationIssue(
                    code="KNOWLEDGE_LIMITATION",
                    message="Response indicates knowledge limitations",
                    level=ValidationLevel.INFO
                )

        return None

    def is_valid(self, content: str) -> bool:
        """Quick check if output is valid."""
        return self.validate(content).is_valid

    def get_safe_content(
        self,
        content: str,
        fallback: str = "I apologize, but I cannot provide a response at this time."
    ) -> str:
        """
        Get validated content or fallback.

        Args:
            content: LLM response
            fallback: Message to use if validation fails

        Returns:
            Original content if valid, fallback otherwise
        """
        result = self.validate(content)
        if result.is_valid:
            return content
        return fallback


def validate_response(content: str) -> ValidationResult:
    """Convenience function for quick validation."""
    validator = OutputValidator()
    return validator.validate(content)
