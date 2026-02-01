"""
Token counting and budget management utilities.

Design Decisions:
- Model-agnostic: Uses tiktoken with fallback to character-based estimation
- Configurable thresholds: Summarization triggers are adjustable
- Message-aware: Understands chat message structure for accurate counting
"""

from typing import Optional
import tiktoken


# Default tokenizer (cl100k_base works for GPT-4, Claude uses similar tokenization)
_DEFAULT_ENCODING = "cl100k_base"

# Fallback: average characters per token when tiktoken unavailable
_CHARS_PER_TOKEN_ESTIMATE = 4


def get_tokenizer(model: Optional[str] = None) -> tiktoken.Encoding:
    """
    Get the appropriate tokenizer for a model.

    Args:
        model: Model name (optional). Falls back to cl100k_base.

    Returns:
        tiktoken Encoding object
    """
    try:
        if model:
            return tiktoken.encoding_for_model(model)
        return tiktoken.get_encoding(_DEFAULT_ENCODING)
    except KeyError:
        # Unknown model, use default
        return tiktoken.get_encoding(_DEFAULT_ENCODING)


def count_tokens(text: str, model: Optional[str] = None) -> int:
    """
    Count tokens in a text string.

    Args:
        text: The text to tokenize
        model: Optional model name for model-specific tokenization

    Returns:
        Number of tokens

    Note:
        Falls back to character-based estimation if tiktoken fails.
    """
    if not text:
        return 0

    try:
        tokenizer = get_tokenizer(model)
        return len(tokenizer.encode(text))
    except Exception:
        # Fallback: estimate based on character count
        return len(text) // _CHARS_PER_TOKEN_ESTIMATE


def count_message_tokens(messages: list[dict], model: Optional[str] = None) -> int:
    """
    Count tokens in a list of chat messages.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        model: Optional model name

    Returns:
        Total token count including message overhead

    Note:
        Adds overhead for message formatting (role, delimiters).
        This is an approximation - exact counts vary by model.
    """
    total = 0

    # Overhead per message: role token + formatting
    message_overhead = 4  # <|role|>\n + content + \n

    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")

        total += count_tokens(role, model)
        total += count_tokens(content, model)
        total += message_overhead

    # Conversation overhead (start/end tokens)
    total += 3

    return total


def calculate_token_budget(
    max_context_tokens: int,
    system_prompt_tokens: int,
    response_buffer: int = 1000
) -> int:
    """
    Calculate available tokens for conversation history.

    Args:
        max_context_tokens: Model's maximum context window
        system_prompt_tokens: Tokens used by system prompt
        response_buffer: Reserved tokens for model response

    Returns:
        Available tokens for conversation history
    """
    available = max_context_tokens - system_prompt_tokens - response_buffer
    return max(0, available)


def should_summarize(
    current_tokens: int,
    max_context_tokens: int,
    threshold_ratio: float = 0.7
) -> bool:
    """
    Determine if conversation should be summarized.

    Args:
        current_tokens: Current token count of conversation
        max_context_tokens: Maximum allowed tokens
        threshold_ratio: Trigger summarization at this percentage (default 70%)

    Returns:
        True if summarization should be triggered

    Design Note:
        We use 70% as default threshold to leave buffer for:
        - New user message
        - System prompt changes
        - Response generation
    """
    threshold = int(max_context_tokens * threshold_ratio)
    return current_tokens >= threshold


def estimate_summary_tokens(original_tokens: int, compression_ratio: float = 0.25) -> int:
    """
    Estimate token count after summarization.

    Args:
        original_tokens: Tokens in original content
        compression_ratio: Expected compression (default 25% of original)

    Returns:
        Estimated tokens after summarization
    """
    return int(original_tokens * compression_ratio)


def tokens_to_chars(tokens: int) -> int:
    """Convert token count to approximate character count."""
    return tokens * _CHARS_PER_TOKEN_ESTIMATE


def chars_to_tokens(chars: int) -> int:
    """Convert character count to approximate token count."""
    return chars // _CHARS_PER_TOKEN_ESTIMATE
