"""
Tests for short-term memory manager.
"""

import pytest
from app.memory.short_term import ShortTermMemoryManager, Message


class TestMessage:
    """Tests for Message dataclass."""

    def test_create_message(self):
        """Should create message with required fields."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.timestamp is not None

    def test_to_dict(self):
        """Should convert to API-compatible dict."""
        msg = Message(role="assistant", content="Hi there")
        d = msg.to_dict()
        assert d == {"role": "assistant", "content": "Hi there"}


class TestShortTermMemoryManager:
    """Tests for ShortTermMemoryManager."""

    def test_init_defaults(self):
        """Should initialize with default values."""
        manager = ShortTermMemoryManager()
        assert manager.max_context_tokens == 8000
        assert manager.summarization_threshold == 0.7
        assert manager.preserve_recent == 4
        assert len(manager.messages) == 0

    def test_init_custom(self):
        """Should accept custom parameters."""
        manager = ShortTermMemoryManager(
            max_context_tokens=4000,
            summarization_threshold=0.5,
            preserve_recent=2
        )
        assert manager.max_context_tokens == 4000
        assert manager.summarization_threshold == 0.5
        assert manager.preserve_recent == 2

    def test_add_message(self):
        """Should add messages and track tokens."""
        manager = ShortTermMemoryManager()
        manager.add_message("user", "Hello")

        assert len(manager.messages) == 1
        assert manager.messages[0].role == "user"
        assert manager.messages[0].content == "Hello"
        assert manager.total_tokens > 0

    def test_add_multiple_messages(self):
        """Should accumulate messages."""
        manager = ShortTermMemoryManager()
        manager.add_message("user", "Hello")
        manager.add_message("assistant", "Hi there!")
        manager.add_message("user", "How are you?")

        assert len(manager.messages) == 3

    def test_messages_immutable(self):
        """Messages property should return copy."""
        manager = ShortTermMemoryManager()
        manager.add_message("user", "Hello")

        messages = manager.messages
        messages.append(Message(role="user", content="Injected"))

        # Original should be unchanged
        assert len(manager.messages) == 1

    def test_needs_summarization_false(self):
        """Should not need summarization with few tokens."""
        manager = ShortTermMemoryManager(max_context_tokens=8000)
        manager.add_message("user", "Hi")

        assert not manager.needs_summarization()

    def test_needs_summarization_true(self):
        """Should need summarization when threshold exceeded."""
        manager = ShortTermMemoryManager(
            max_context_tokens=100,
            summarization_threshold=0.7
        )
        # Add enough messages to exceed threshold
        for i in range(10):
            manager.add_message("user", "x" * 50)

        assert manager.needs_summarization()

    def test_has_summary_false_initially(self):
        """Should not have summary initially."""
        manager = ShortTermMemoryManager()
        assert not manager.has_summary

    def test_get_context_messages_no_summary(self):
        """Should return messages without summary prefix."""
        manager = ShortTermMemoryManager()
        manager.add_message("user", "Hello")
        manager.add_message("assistant", "Hi")

        context = manager.get_context_messages()
        assert len(context) == 2
        assert context[0] == {"role": "user", "content": "Hello"}
        assert context[1] == {"role": "assistant", "content": "Hi"}

    def test_clear(self):
        """Should clear all messages and reset state."""
        manager = ShortTermMemoryManager()
        manager.add_message("user", "Hello")
        manager.add_message("assistant", "Hi")

        manager.clear()

        assert len(manager.messages) == 0
        assert manager.total_tokens == 0
        assert not manager.has_summary

    def test_get_stats(self):
        """Should return useful statistics."""
        manager = ShortTermMemoryManager(max_context_tokens=1000)
        manager.add_message("user", "Hello")

        stats = manager.get_stats()

        assert "message_count" in stats
        assert "total_tokens" in stats
        assert "max_tokens" in stats
        assert "usage_percent" in stats
        assert "has_summary" in stats
        assert stats["message_count"] == 1
        assert stats["max_tokens"] == 1000

    def test_get_messages_for_summarization(self):
        """Should return oldest N messages."""
        manager = ShortTermMemoryManager()
        manager.add_message("user", "Message 1")
        manager.add_message("assistant", "Response 1")
        manager.add_message("user", "Message 2")
        manager.add_message("assistant", "Response 2")

        oldest = manager.get_messages_for_summarization(2)

        assert len(oldest) == 2
        assert oldest[0]["content"] == "Message 1"
        assert oldest[1]["content"] == "Response 1"


class TestSummarization:
    """Tests for summarization functionality."""

    @pytest.mark.asyncio
    async def test_summarize_if_needed_not_needed(self):
        """Should not summarize when under threshold."""
        manager = ShortTermMemoryManager(max_context_tokens=8000)
        manager.add_message("user", "Hello")

        async def mock_summarizer(messages):
            return "Summary"

        result = await manager.summarize_if_needed(mock_summarizer)
        assert result is False

    @pytest.mark.asyncio
    async def test_summarize_if_needed_not_enough_messages(self):
        """Should not summarize if fewer messages than preserve_recent."""
        manager = ShortTermMemoryManager(
            max_context_tokens=50,
            preserve_recent=4
        )
        manager.add_message("user", "x" * 100)  # Exceeds threshold
        manager.add_message("assistant", "y" * 100)

        async def mock_summarizer(messages):
            return "Summary"

        result = await manager.summarize_if_needed(mock_summarizer)
        assert result is False  # Only 2 messages, need > 4

    @pytest.mark.asyncio
    async def test_summarize_if_needed_performs_summarization(self):
        """Should summarize when conditions met."""
        manager = ShortTermMemoryManager(
            max_context_tokens=100,
            summarization_threshold=0.3,
            preserve_recent=2
        )

        # Add many messages to exceed threshold
        for i in range(6):
            manager.add_message("user", f"Message {i}")
            manager.add_message("assistant", f"Response {i}")

        async def mock_summarizer(messages):
            return "This is a test summary"

        result = await manager.summarize_if_needed(mock_summarizer)

        # Check result
        assert result is True
        assert manager.has_summary
        assert len(manager.messages) == 2  # Only preserve_recent kept

    @pytest.mark.asyncio
    async def test_get_context_messages_with_summary(self):
        """Should include summary in context when present."""
        manager = ShortTermMemoryManager(
            max_context_tokens=100,
            summarization_threshold=0.3,
            preserve_recent=2
        )

        for i in range(6):
            manager.add_message("user", f"Message {i}")
            manager.add_message("assistant", f"Response {i}")

        async def mock_summarizer(messages):
            return "Previous conversation about various topics"

        await manager.summarize_if_needed(mock_summarizer)

        context = manager.get_context_messages()

        # First message should be the summary
        assert context[0]["role"] == "system"
        assert "summary" in context[0]["content"].lower()
        # Remaining should be preserved messages
        assert len(context) == 3  # 1 summary + 2 preserved
