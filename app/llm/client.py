"""
LLM Client Wrapper.

Design Decisions:
- Single point of contact: All LLM calls must go through here
- Token tracking: Every call logs input/output tokens
- Safety first: System prompt always takes precedence
- Provider-agnostic: Abstract base with OpenAI/Anthropic implementations
- Async-native: Non-blocking API calls
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import time
import logging

from app.llm.token_utils import count_tokens, count_message_tokens


logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from an LLM call with metadata."""
    content: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    model: str
    finish_reason: Optional[str] = None


@dataclass
class UsageStats:
    """Cumulative usage statistics."""
    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_ms: float = 0

    def record(self, response: LLMResponse) -> None:
        """Record usage from a response."""
        self.total_requests += 1
        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens
        self.total_latency_ms += response.latency_ms


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, model: str, max_tokens: int = 1000):
        """
        Initialize LLM client.

        Args:
            model: Model identifier
            max_tokens: Default max tokens for responses
        """
        self.model = model
        self.max_tokens = max_tokens
        self.usage_stats = UsageStats()

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            system_prompt: System instructions (always takes precedence)
            user_message: User's message
            max_tokens: Override default max tokens
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    async def chat(
        self,
        system_prompt: str,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7
    ) -> LLMResponse:
        """
        Multi-turn chat with the LLM.

        Args:
            system_prompt: System instructions
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Override default max tokens
            temperature: Sampling temperature

        Returns:
            LLMResponse with content and metadata
        """
        pass

    def get_usage_stats(self) -> dict:
        """Get current usage statistics."""
        return {
            "total_requests": self.usage_stats.total_requests,
            "total_input_tokens": self.usage_stats.total_input_tokens,
            "total_output_tokens": self.usage_stats.total_output_tokens,
            "total_tokens": (
                self.usage_stats.total_input_tokens +
                self.usage_stats.total_output_tokens
            ),
            "avg_latency_ms": (
                self.usage_stats.total_latency_ms / self.usage_stats.total_requests
                if self.usage_stats.total_requests > 0 else 0
            )
        }


class OpenAIClient(LLMClient):
    """OpenAI API client implementation."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_tokens: int = 1000
    ):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key
            model: Model name (default: gpt-4o-mini)
            max_tokens: Default max tokens
        """
        super().__init__(model, max_tokens)
        self.api_key = api_key
        self._client = None

    async def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7
    ) -> str:
        """Generate a single response."""
        messages = [{"role": "user", "content": user_message}]
        response = await self.chat(
            system_prompt=system_prompt,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.content

    async def chat(
        self,
        system_prompt: str,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7
    ) -> LLMResponse:
        """Multi-turn chat."""
        client = await self._get_client()

        # Build messages with system prompt first (ensures precedence)
        full_messages = [{"role": "system", "content": system_prompt}]
        full_messages.extend(messages)

        # Log request
        input_tokens = count_message_tokens(full_messages, self.model)
        logger.info(f"LLM request: model={self.model}, input_tokens={input_tokens}")

        start_time = time.time()

        response = await client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature
        )

        latency_ms = (time.time() - start_time) * 1000

        # Extract response data
        content = response.choices[0].message.content or ""
        output_tokens = response.usage.completion_tokens if response.usage else count_tokens(content)
        actual_input_tokens = response.usage.prompt_tokens if response.usage else input_tokens

        result = LLMResponse(
            content=content,
            input_tokens=actual_input_tokens,
            output_tokens=output_tokens,
            total_tokens=actual_input_tokens + output_tokens,
            latency_ms=latency_ms,
            model=self.model,
            finish_reason=response.choices[0].finish_reason
        )

        # Record usage
        self.usage_stats.record(result)

        logger.info(
            f"LLM response: output_tokens={output_tokens}, "
            f"latency_ms={latency_ms:.0f}, finish={result.finish_reason}"
        )

        return result


class AnthropicClient(LLMClient):
    """Anthropic Claude API client implementation."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-haiku-20240307",
        max_tokens: int = 1000
    ):
        """
        Initialize Anthropic client.

        Args:
            api_key: Anthropic API key
            model: Model name
            max_tokens: Default max tokens
        """
        super().__init__(model, max_tokens)
        self.api_key = api_key
        self._client = None

    async def _get_client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(api_key=self.api_key)
        return self._client

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7
    ) -> str:
        """Generate a single response."""
        messages = [{"role": "user", "content": user_message}]
        response = await self.chat(
            system_prompt=system_prompt,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.content

    async def chat(
        self,
        system_prompt: str,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7
    ) -> LLMResponse:
        """Multi-turn chat."""
        client = await self._get_client()

        # Anthropic uses separate system parameter
        input_tokens = count_tokens(system_prompt) + count_message_tokens(messages)
        logger.info(f"LLM request: model={self.model}, input_tokens={input_tokens}")

        start_time = time.time()

        response = await client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=messages,
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature
        )

        latency_ms = (time.time() - start_time) * 1000

        # Extract response data
        content = response.content[0].text if response.content else ""
        output_tokens = response.usage.output_tokens
        actual_input_tokens = response.usage.input_tokens

        result = LLMResponse(
            content=content,
            input_tokens=actual_input_tokens,
            output_tokens=output_tokens,
            total_tokens=actual_input_tokens + output_tokens,
            latency_ms=latency_ms,
            model=self.model,
            finish_reason=response.stop_reason
        )

        # Record usage
        self.usage_stats.record(result)

        logger.info(
            f"LLM response: output_tokens={output_tokens}, "
            f"latency_ms={latency_ms:.0f}, finish={result.finish_reason}"
        )

        return result
