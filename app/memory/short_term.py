"""
Short-term memory manager.

Design Decisions:
- Token-aware: Always tracks token usage
- Lazy summarization: Only summarizes when threshold exceeded
- Preserves recent context: Keeps last N messages intact during summarization
- Thread-safe ready: Uses dataclass for immutable message storage
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable
from datetime import datetime

from app.llm.token_utils import count_message_tokens, should_summarize


@dataclass
class Message:
    """Immutable message container."""
    role: str  # 'user', 'assistant', or 'system'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    token_count: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to API-compatible dict."""
        return {"role": self.role, "content": self.content}


class ShortTermMemoryManager:
    """
    Manages conversation history with automatic summarization.

    Key responsibilities:
    - Store conversation messages
    - Track total token usage
    - Trigger summarization when threshold exceeded
    - Replace older messages with summary
    """

    def __init__(
        self,
        max_context_tokens: int = 8000,
        summarization_threshold: float = 0.7,
        preserve_recent: int = 4,
        model: Optional[str] = None
    ):
        """
        Initialize short-term memory.

        Args:
            max_context_tokens: Maximum tokens allowed in context
            summarization_threshold: Trigger at this % of max (default 70%)
            preserve_recent: Number of recent messages to keep during summarization
            model: Model name for accurate token counting
        """
        self.max_context_tokens = max_context_tokens
        self.summarization_threshold = summarization_threshold
        self.preserve_recent = preserve_recent
        self.model = model

        self._messages: list[Message] = []
        self._summary: Optional[str] = None
        self._total_tokens: int = 0

    @property
    def messages(self) -> list[Message]:
        """Get all messages (read-only view)."""
        return self._messages.copy()

    @property
    def total_tokens(self) -> int:
        """Get current token count."""
        return self._total_tokens

    @property
    def has_summary(self) -> bool:
        """Check if conversation has been summarized."""
        return self._summary is not None

    def add_message(self, role: str, content: str) -> None:
        """
        Add a new message to memory.

        Args:
            role: Message role ('user' or 'assistant')
            content: Message content
        """
        message = Message(role=role, content=content)
        self._messages.append(message)
        self._recalculate_tokens()

    def needs_summarization(self) -> bool:
        """Check if memory exceeds summarization threshold."""
        return should_summarize(
            current_tokens=self._total_tokens,
            max_context_tokens=self.max_context_tokens,
            threshold_ratio=self.summarization_threshold
        )

    async def summarize_if_needed(
        self,
        summarizer: Callable[[list[dict]], Awaitable[str]]
    ) -> bool:
        """
        Summarize older messages if threshold exceeded.

        Args:
            summarizer: Async function that takes messages and returns summary

        Returns:
            True if summarization was performed

        Design Note:
            We keep the last `preserve_recent` messages intact.
            Everything before gets summarized and replaced.
        """
        if not self.needs_summarization():
            return False

        if len(self._messages) <= self.preserve_recent:
            # Not enough messages to summarize
            return False

        # Split messages: older (to summarize) vs recent (to keep)
        split_index = len(self._messages) - self.preserve_recent
        older_messages = self._messages[:split_index]
        recent_messages = self._messages[split_index:]

        # Convert to dicts for summarizer
        messages_to_summarize = [m.to_dict() for m in older_messages]

        # Include previous summary if exists
        if self._summary:
            messages_to_summarize.insert(0, {
                "role": "system",
                "content": f"Previous conversation summary: {self._summary}"
            })

        # Generate new summary
        new_summary = await summarizer(messages_to_summarize)

        # Update state
        self._summary = new_summary
        self._messages = recent_messages
        self._recalculate_tokens()

        return True

    def get_context_messages(self) -> list[dict]:
        """
        Get messages formatted for LLM context.

        Returns:
            List of message dicts, with summary prepended if exists
        """
        result = []

        # Add summary as system context if exists
        if self._summary:
            result.append({
                "role": "system",
                "content": f"Conversation summary: {self._summary}"
            })

        # Add all messages
        for msg in self._messages:
            result.append(msg.to_dict())

        return result

    def get_messages_for_summarization(self, count: int) -> list[dict]:
        """
        Get oldest N messages for summarization.

        Args:
            count: Number of messages to get

        Returns:
            List of message dicts
        """
        messages = self._messages[:count]
        return [m.to_dict() for m in messages]

    def clear(self) -> None:
        """Clear all messages and summary."""
        self._messages = []
        self._summary = None
        self._total_tokens = 0

    def _recalculate_tokens(self) -> None:
        """Recalculate total token count."""
        messages_as_dicts = [m.to_dict() for m in self._messages]

        # Count message tokens
        self._total_tokens = count_message_tokens(messages_as_dicts, self.model)

        # Add summary tokens if exists
        if self._summary:
            from app.llm.token_utils import count_tokens
            self._total_tokens += count_tokens(self._summary, self.model)

    def get_stats(self) -> dict:
        """Get memory statistics for logging/debugging."""
        return {
            "message_count": len(self._messages),
            "total_tokens": self._total_tokens,
            "max_tokens": self.max_context_tokens,
            "usage_percent": round(self._total_tokens / self.max_context_tokens * 100, 1),
            "has_summary": self.has_summary,
            "threshold_percent": self.summarization_threshold * 100
        }
