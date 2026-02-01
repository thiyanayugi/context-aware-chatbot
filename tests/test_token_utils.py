"""
Tests for token counting utilities.
"""

import pytest
from app.llm.token_utils import (
    count_tokens,
    count_message_tokens,
    calculate_token_budget,
    should_summarize,
    estimate_summary_tokens,
    tokens_to_chars,
    chars_to_tokens,
    get_tokenizer,
)


class TestCountTokens:
    """Tests for count_tokens function."""

    def test_empty_string(self):
        """Empty string should return 0 tokens."""
        assert count_tokens("") == 0

    def test_simple_text(self):
        """Basic text should return positive token count."""
        tokens = count_tokens("Hello, world!")
        assert tokens > 0
        assert tokens < 10  # Simple greeting shouldn't be many tokens

    def test_longer_text(self):
        """Longer text should have proportionally more tokens."""
        short = count_tokens("Hi")
        long = count_tokens("This is a much longer sentence with many more words.")
        assert long > short

    def test_with_model_name(self):
        """Should work with model name specified."""
        tokens = count_tokens("Hello", model="gpt-4")
        assert tokens > 0

    def test_unknown_model_fallback(self):
        """Unknown model should fall back to default tokenizer."""
        tokens = count_tokens("Hello", model="unknown-model-xyz")
        assert tokens > 0


class TestCountMessageTokens:
    """Tests for count_message_tokens function."""

    def test_empty_list(self):
        """Empty message list should return minimal overhead tokens."""
        tokens = count_message_tokens([])
        assert tokens >= 0

    def test_single_message(self, sample_messages):
        """Single message should have reasonable token count."""
        single = [sample_messages[0]]
        tokens = count_message_tokens(single)
        assert tokens > 0
        assert tokens < 50

    def test_multiple_messages(self, sample_messages):
        """Multiple messages should have cumulative tokens."""
        single_tokens = count_message_tokens([sample_messages[0]])
        all_tokens = count_message_tokens(sample_messages)
        assert all_tokens > single_tokens

    def test_message_overhead(self):
        """Messages should include overhead for formatting."""
        # Two identical messages should have similar overhead each
        msg = {"role": "user", "content": "test"}
        one_msg = count_message_tokens([msg])
        two_msgs = count_message_tokens([msg, msg])
        # Two messages should be roughly 2x one message (with some overhead)
        assert two_msgs > one_msg


class TestCalculateTokenBudget:
    """Tests for calculate_token_budget function."""

    def test_basic_calculation(self):
        """Budget should be max minus system and response buffer."""
        budget = calculate_token_budget(
            max_context_tokens=8000,
            system_prompt_tokens=500,
            response_buffer=1000
        )
        assert budget == 6500

    def test_zero_system_prompt(self):
        """Zero system prompt should give more budget."""
        budget = calculate_token_budget(
            max_context_tokens=8000,
            system_prompt_tokens=0,
            response_buffer=1000
        )
        assert budget == 7000

    def test_negative_budget_floors_to_zero(self):
        """If budget would be negative, should return 0."""
        budget = calculate_token_budget(
            max_context_tokens=1000,
            system_prompt_tokens=800,
            response_buffer=500
        )
        assert budget == 0


class TestShouldSummarize:
    """Tests for should_summarize function."""

    def test_below_threshold(self):
        """Below threshold should not trigger summarization."""
        assert not should_summarize(
            current_tokens=500,
            max_context_tokens=1000,
            threshold_ratio=0.7
        )

    def test_at_threshold(self):
        """At threshold should trigger summarization."""
        assert should_summarize(
            current_tokens=700,
            max_context_tokens=1000,
            threshold_ratio=0.7
        )

    def test_above_threshold(self):
        """Above threshold should trigger summarization."""
        assert should_summarize(
            current_tokens=800,
            max_context_tokens=1000,
            threshold_ratio=0.7
        )

    def test_custom_threshold(self):
        """Custom threshold should be respected."""
        assert should_summarize(
            current_tokens=500,
            max_context_tokens=1000,
            threshold_ratio=0.5
        )


class TestEstimateSummaryTokens:
    """Tests for estimate_summary_tokens function."""

    def test_default_compression(self):
        """Default compression should be 25%."""
        estimated = estimate_summary_tokens(1000)
        assert estimated == 250

    def test_custom_compression(self):
        """Custom compression ratio should work."""
        estimated = estimate_summary_tokens(1000, compression_ratio=0.5)
        assert estimated == 500


class TestConversions:
    """Tests for token/char conversion functions."""

    def test_tokens_to_chars(self):
        """Token to char conversion."""
        chars = tokens_to_chars(100)
        assert chars == 400  # 4 chars per token default

    def test_chars_to_tokens(self):
        """Char to token conversion."""
        tokens = chars_to_tokens(400)
        assert tokens == 100

    def test_round_trip(self):
        """Converting back and forth should be consistent."""
        original = 100
        chars = tokens_to_chars(original)
        back = chars_to_tokens(chars)
        assert back == original


class TestGetTokenizer:
    """Tests for get_tokenizer function."""

    def test_default_encoding(self):
        """Default encoding should work."""
        tokenizer = get_tokenizer()
        assert tokenizer is not None

    def test_with_model(self):
        """Model-specific tokenizer should work."""
        tokenizer = get_tokenizer("gpt-4")
        assert tokenizer is not None

    def test_unknown_model(self):
        """Unknown model should fall back to default."""
        tokenizer = get_tokenizer("unknown-model")
        assert tokenizer is not None
