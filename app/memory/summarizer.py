"""
Conversation summarizer.

Design Decisions:
- Focused prompts: Instructs LLM to preserve facts, drop fluff
- Token-aware: Targets specific compression ratio
- Incremental: Can include previous summary for continuity
- Async: Non-blocking LLM calls
"""

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.llm.client import LLMClient


# System prompt for summarization
SUMMARIZATION_PROMPT = """You are a conversation summarizer. Your task is to create a concise summary of the conversation that preserves:

1. Key facts and information shared
2. Important decisions made
3. User preferences and requirements mentioned
4. Any ongoing tasks or action items
5. Context needed to continue the conversation

Rules:
- Be concise but complete
- Use bullet points for clarity
- Preserve specific details (names, numbers, dates)
- Remove pleasantries, greetings, and filler
- Keep the summary under 200 words
- Write in third person (e.g., "The user asked about...")

Output only the summary, no preamble."""


class ConversationSummarizer:
    """
    Summarizes conversation history using an LLM.

    Uses a specialized prompt to compress conversations while
    preserving important context for future interactions.
    """

    def __init__(
        self,
        llm_client: "LLMClient",
        max_summary_tokens: int = 500
    ):
        """
        Initialize the summarizer.

        Args:
            llm_client: LLM client for generating summaries
            max_summary_tokens: Maximum tokens for generated summary
        """
        self.llm_client = llm_client
        self.max_summary_tokens = max_summary_tokens

    async def summarize(
        self,
        messages: list[dict],
        previous_summary: Optional[str] = None
    ) -> str:
        """
        Summarize a list of conversation messages.

        Args:
            messages: List of message dicts to summarize
            previous_summary: Optional previous summary to include

        Returns:
            Compressed summary of the conversation

        Note:
            If a previous summary exists, it's included for continuity.
            The LLM will merge old and new context.
        """
        # Build the content to summarize
        content_parts = []

        if previous_summary:
            content_parts.append(f"Previous summary:\n{previous_summary}\n")

        content_parts.append("New messages to incorporate:\n")

        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            content_parts.append(f"{role.upper()}: {content}")

        content_to_summarize = "\n".join(content_parts)

        # Call LLM for summarization
        summary = await self.llm_client.generate(
            system_prompt=SUMMARIZATION_PROMPT,
            user_message=content_to_summarize,
            max_tokens=self.max_summary_tokens
        )

        return summary.strip()

    async def summarize_for_storage(
        self,
        messages: list[dict],
        topic_hint: Optional[str] = None
    ) -> dict:
        """
        Create a summary optimized for long-term storage.

        Args:
            messages: Messages to summarize
            topic_hint: Optional hint about conversation topic

        Returns:
            Dict with summary and metadata for vector storage
        """
        summary = await self.summarize(messages)

        # Extract key topics for better retrieval
        topics = await self._extract_topics(messages, topic_hint)

        return {
            "summary": summary,
            "topics": topics,
            "message_count": len(messages),
            "first_message_role": messages[0].get("role") if messages else None
        }

    async def _extract_topics(
        self,
        messages: list[dict],
        topic_hint: Optional[str] = None
    ) -> list[str]:
        """
        Extract key topics from messages for retrieval.

        Args:
            messages: Messages to analyze
            topic_hint: Optional pre-known topic

        Returns:
            List of topic keywords
        """
        if topic_hint:
            return [topic_hint]

        # Simple extraction: use LLM to identify topics
        content = "\n".join(
            f"{m.get('role')}: {m.get('content')}"
            for m in messages[:5]  # Limit to first 5 for efficiency
        )

        topic_prompt = """Extract 3-5 key topics from this conversation as a comma-separated list.
Topics should be single words or short phrases.
Output only the topics, nothing else."""

        topics_str = await self.llm_client.generate(
            system_prompt=topic_prompt,
            user_message=content,
            max_tokens=50
        )

        # Parse comma-separated topics
        topics = [t.strip() for t in topics_str.split(",") if t.strip()]
        return topics[:5]  # Limit to 5 topics


def create_summarizer_callback(summarizer: ConversationSummarizer):
    """
    Create a callback function compatible with ShortTermMemoryManager.

    Args:
        summarizer: ConversationSummarizer instance

    Returns:
        Async function that takes messages and returns summary string
    """
    async def summarize_callback(messages: list[dict]) -> str:
        return await summarizer.summarize(messages)

    return summarize_callback
