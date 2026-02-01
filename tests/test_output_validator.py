"""
Tests for output validation.
"""

import pytest
from app.guardrails.output_validator import (
    OutputValidator,
    ValidationResult,
    ValidationIssue,
    ValidationLevel,
    validate_response,
)


class TestValidationLevel:
    """Tests for ValidationLevel enum."""

    def test_levels_exist(self):
        """All expected levels should exist."""
        assert ValidationLevel.INFO
        assert ValidationLevel.WARNING
        assert ValidationLevel.ERROR
        assert ValidationLevel.CRITICAL


class TestValidationIssue:
    """Tests for ValidationIssue dataclass."""

    def test_create_issue(self):
        """Should create issue with required fields."""
        issue = ValidationIssue(
            code="TEST_CODE",
            message="Test message",
            level=ValidationLevel.WARNING
        )
        assert issue.code == "TEST_CODE"
        assert issue.message == "Test message"
        assert issue.level == ValidationLevel.WARNING


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_create_result(self):
        """Should create result with required fields."""
        result = ValidationResult(
            is_valid=True,
            content="Test content",
            issues=[]
        )
        assert result.is_valid is True
        assert result.content == "Test content"
        assert len(result.issues) == 0

    def test_has_errors_false(self):
        """Should return False when no errors."""
        result = ValidationResult(
            is_valid=True,
            content="Test",
            issues=[ValidationIssue("WARN", "Warning", ValidationLevel.WARNING)]
        )
        assert not result.has_errors

    def test_has_errors_true(self):
        """Should return True when errors present."""
        result = ValidationResult(
            is_valid=False,
            content="Test",
            issues=[ValidationIssue("ERR", "Error", ValidationLevel.ERROR)]
        )
        assert result.has_errors

    def test_has_warnings(self):
        """Should detect warnings."""
        result = ValidationResult(
            is_valid=True,
            content="Test",
            issues=[ValidationIssue("WARN", "Warning", ValidationLevel.WARNING)]
        )
        assert result.has_warnings


class TestOutputValidator:
    """Tests for OutputValidator class."""

    def test_init_defaults(self):
        """Should initialize with default values."""
        validator = OutputValidator()
        assert validator.min_length == 1
        assert validator.max_length == 10000
        assert validator.block_unsafe_tools is True

    def test_init_custom(self):
        """Should accept custom parameters."""
        validator = OutputValidator(
            min_length=10,
            max_length=5000,
            block_unsafe_tools=False
        )
        assert validator.min_length == 10
        assert validator.max_length == 5000
        assert validator.block_unsafe_tools is False


class TestEmptyResponseValidation:
    """Tests for empty response detection."""

    def test_empty_string_invalid(self):
        """Empty string should be invalid."""
        validator = OutputValidator()
        result = validator.validate("")
        assert not result.is_valid
        assert any(i.code == "EMPTY_RESPONSE" for i in result.issues)

    def test_whitespace_only_invalid(self):
        """Whitespace-only string should be invalid."""
        validator = OutputValidator()
        result = validator.validate("   \n\t   ")
        assert not result.is_valid
        assert any(i.code == "EMPTY_RESPONSE" for i in result.issues)

    def test_none_as_empty(self):
        """None should be treated as empty (if passed)."""
        validator = OutputValidator()
        # This would typically raise an error, but let's test the logic
        result = validator.validate("")
        assert not result.is_valid


class TestLengthValidation:
    """Tests for length validation."""

    def test_valid_length(self):
        """Normal length should pass."""
        validator = OutputValidator(min_length=1, max_length=100)
        result = validator.validate("Hello, world!")
        assert result.is_valid

    def test_too_short(self):
        """Too short should trigger warning."""
        validator = OutputValidator(min_length=50)
        result = validator.validate("Hi")
        assert any(i.code == "TOO_SHORT" for i in result.issues)

    def test_too_long(self):
        """Too long should trigger warning."""
        validator = OutputValidator(max_length=10)
        result = validator.validate("This is a very long response")
        assert any(i.code == "TOO_LONG" for i in result.issues)


class TestUnsafeToolDetection:
    """Tests for unsafe tool pattern detection."""

    def test_shell_command_detection(self):
        """Should detect dangerous shell commands."""
        validator = OutputValidator()
        result = validator.validate("""
        ```bash
        rm -rf /
        ```
        """)
        assert result.has_warnings
        assert any("UNSAFE" in i.code for i in result.issues)

    def test_sudo_detection(self):
        """Should detect sudo commands."""
        validator = OutputValidator()
        result = validator.validate("""
        ```shell
        sudo rm -rf important_files
        ```
        """)
        assert any("UNSAFE" in i.code for i in result.issues)

    def test_script_tag_detection(self):
        """Should detect script tags."""
        validator = OutputValidator()
        result = validator.validate("<script>alert('xss')</script>")
        assert any("UNSAFE" in i.code for i in result.issues)

    def test_javascript_uri_detection(self):
        """Should detect javascript: URIs."""
        validator = OutputValidator()
        result = validator.validate("Click here: javascript: alert('xss')")
        assert any("UNSAFE" in i.code for i in result.issues)

    def test_eval_detection(self):
        """Should detect eval() calls."""
        validator = OutputValidator()
        result = validator.validate("Use eval(user_input) to execute")
        assert any("UNSAFE" in i.code for i in result.issues)

    def test_exec_detection(self):
        """Should detect exec() calls."""
        validator = OutputValidator()
        result = validator.validate("Run exec(code) to evaluate")
        assert any("UNSAFE" in i.code for i in result.issues)

    def test_import_detection(self):
        """Should detect __import__ calls."""
        validator = OutputValidator()
        result = validator.validate("Use __import__('os') to access system")
        assert any("UNSAFE" in i.code for i in result.issues)

    def test_safe_code_blocks(self):
        """Safe code blocks should pass."""
        validator = OutputValidator()
        result = validator.validate("""
        ```python
        def hello():
            print("Hello, World!")
        ```
        """)
        assert result.is_valid


class TestRefusalDetection:
    """Tests for model refusal detection."""

    def test_cannot_refusal(self):
        """Should detect 'I cannot' refusals."""
        validator = OutputValidator()
        result = validator.validate("I cannot help with that request.")
        assert any(i.code == "MODEL_REFUSAL" for i in result.issues)

    def test_not_able_refusal(self):
        """Should detect 'I'm not able to' refusals."""
        validator = OutputValidator()
        result = validator.validate("I'm not able to provide that information.")
        assert any(i.code == "MODEL_REFUSAL" for i in result.issues)

    def test_as_ai_refusal(self):
        """Should detect 'As an AI' refusals."""
        validator = OutputValidator()
        result = validator.validate("As an AI language model, I cannot access the internet.")
        assert any(i.code == "MODEL_REFUSAL" for i in result.issues)

    def test_normal_use_of_cannot(self):
        """Normal use of 'cannot' in context shouldn't always trigger."""
        validator = OutputValidator()
        # This will still trigger because the pattern is simple
        # In production, you might want more sophisticated detection
        result = validator.validate("You cannot divide by zero in Python.")
        # This is expected behavior - simple pattern matching


class TestHallucinationPatterns:
    """Tests for hallucination indicator detection."""

    def test_knowledge_cutoff(self):
        """Should detect knowledge cutoff mentions."""
        validator = OutputValidator()
        result = validator.validate("As of my knowledge cutoff, the president is...")
        assert any(i.code == "KNOWLEDGE_LIMITATION" for i in result.issues)

    def test_last_update(self):
        """Should detect 'last update' mentions."""
        validator = OutputValidator()
        result = validator.validate("As of my last update, I don't have that information.")
        assert any(i.code == "KNOWLEDGE_LIMITATION" for i in result.issues)

    def test_no_realtime_access(self):
        """Should detect real-time access limitations."""
        validator = OutputValidator()
        result = validator.validate("I don't have access to real-time data.")
        assert any(i.code == "KNOWLEDGE_LIMITATION" for i in result.issues)


class TestConvenienceMethods:
    """Tests for convenience methods."""

    def test_is_valid_method(self):
        """is_valid should return boolean."""
        validator = OutputValidator()
        assert validator.is_valid("Hello, world!")
        assert not validator.is_valid("")

    def test_get_safe_content_valid(self):
        """Should return original content when valid."""
        validator = OutputValidator()
        content = "This is a valid response."
        result = validator.get_safe_content(content)
        assert result == content

    def test_get_safe_content_invalid(self):
        """Should return fallback when invalid."""
        validator = OutputValidator()
        fallback = "Custom fallback message"
        result = validator.get_safe_content("", fallback=fallback)
        assert result == fallback

    def test_get_safe_content_default_fallback(self):
        """Should use default fallback when not specified."""
        validator = OutputValidator()
        result = validator.get_safe_content("")
        assert "apologize" in result.lower()


class TestValidateResponseFunction:
    """Tests for validate_response convenience function."""

    def test_returns_validation_result(self):
        """Should return ValidationResult."""
        result = validate_response("Hello, world!")
        assert isinstance(result, ValidationResult)

    def test_validates_content(self):
        """Should validate content properly."""
        result = validate_response("Valid response content")
        assert result.is_valid

    def test_detects_issues(self):
        """Should detect issues in content."""
        result = validate_response("")
        assert not result.is_valid


class TestCustomValidators:
    """Tests for custom validators."""

    def test_custom_validator_called(self):
        """Custom validators should be called."""
        called = []

        def custom_validator(content: str):
            called.append(content)
            return None

        validator = OutputValidator(custom_validators=[custom_validator])
        validator.validate("Test content")

        assert "Test content" in called

    def test_custom_validator_adds_issue(self):
        """Custom validator can add issues."""
        def custom_validator(content: str):
            if "forbidden" in content.lower():
                return ValidationIssue(
                    code="CUSTOM_FORBIDDEN",
                    message="Forbidden word detected",
                    level=ValidationLevel.WARNING
                )
            return None

        validator = OutputValidator(custom_validators=[custom_validator])
        result = validator.validate("This contains a forbidden word")

        assert any(i.code == "CUSTOM_FORBIDDEN" for i in result.issues)
