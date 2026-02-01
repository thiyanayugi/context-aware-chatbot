"""
Main entry point for the Context-Aware Chatbot.

Design Decisions:
- Orchestrator pattern: Coordinates all components
- Async-first: Non-blocking I/O throughout
- Graceful degradation: Continues even if some components fail
- Observable: Comprehensive logging at each step
"""

import asyncio
from dataclasses import dataclass
from typing import Optional
import logging

from app.config import Settings, get_settings, LLMProvider, EmbeddingProvider
from app.utils.logger import setup_logging, get_logger, log_token_usage

from app.llm.client import LLMClient, OpenAIClient, AnthropicClient
from app.llm.token_utils import count_tokens, calculate_token_budget

from app.memory.short_term import ShortTermMemoryManager
from app.memory.long_term import LongTermMemoryManager
from app.memory.summarizer import ConversationSummarizer, create_summarizer_callback

from app.rag.embeddings import EmbeddingProvider as EmbeddingProviderBase
from app.rag.embeddings import OpenAIEmbeddings, SentenceTransformerEmbeddings
from app.rag.vector_store import FAISSVectorStore

from app.guardrails.input_filter import InputFilter
from app.guardrails.output_validator import OutputValidator


logger = get_logger(__name__)


@dataclass
class ChatResponse:
    """Response from a chat interaction."""
    content: str
    input_tokens: int
    output_tokens: int
    was_summarized: bool
    long_term_context_used: bool
    input_filtered: bool


class ContextAwareChatbot:
    """
    Main chatbot orchestrator.

    Implements the full chat flow:
    1. Receive user input
    2. Filter input (guardrails)
    3. Retrieve relevant long-term memory
    4. Add short-term memory context
    5. Check token budget
    6. Summarize if needed
    7. Send to LLM
    8. Validate output
    9. Store conversation
    """

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the chatbot with all components.

        Args:
            settings: Application settings (uses defaults if None)
        """
        self.settings = settings or get_settings()
        self._initialized = False

        # Components (initialized lazily)
        self._llm_client: Optional[LLMClient] = None
        self._short_term_memory: Optional[ShortTermMemoryManager] = None
        self._long_term_memory: Optional[LongTermMemoryManager] = None
        self._summarizer: Optional[ConversationSummarizer] = None
        self._input_filter: Optional[InputFilter] = None
        self._output_validator: Optional[OutputValidator] = None

    async def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return

        logger.info("Initializing chatbot components...")

        # Setup logging
        setup_logging(
            level=self.settings.log_level,
            format_type=self.settings.log_format
        )

        # Initialize LLM client
        self._llm_client = self._create_llm_client()
        logger.info(f"LLM client initialized: {self.settings.llm_provider}")

        # Initialize memory components
        self._short_term_memory = ShortTermMemoryManager(
            max_context_tokens=self.settings.max_context_tokens,
            summarization_threshold=self.settings.summarization_threshold,
            preserve_recent=self.settings.preserve_recent_messages,
            model=self.settings.llm_model
        )

        # Initialize embedding provider
        embedding_provider = self._create_embedding_provider()

        # Initialize vector store
        vector_store = FAISSVectorStore(
            dimension=self.settings.embedding_dimension
        )

        # Initialize long-term memory
        self._long_term_memory = LongTermMemoryManager(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            min_relevance_score=self.settings.min_relevance_score
        )

        # Try to load existing memory
        try:
            await self._long_term_memory.load(self.settings.long_term_memory_path)
            logger.info("Loaded existing long-term memory")
        except Exception as e:
            logger.info(f"No existing memory found, starting fresh: {e}")

        # Initialize summarizer
        self._summarizer = ConversationSummarizer(
            llm_client=self._llm_client
        )

        # Initialize guardrails
        self._input_filter = InputFilter(
            max_length=self.settings.max_input_length
        )
        self._output_validator = OutputValidator(
            max_length=self.settings.max_output_length
        )

        self._initialized = True
        logger.info("Chatbot initialization complete")

    def _create_llm_client(self) -> LLMClient:
        """Create the appropriate LLM client."""
        if self.settings.llm_provider == LLMProvider.OPENAI:
            if not self.settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY required for OpenAI provider")
            return OpenAIClient(
                api_key=self.settings.openai_api_key,
                model=self.settings.llm_model,
                max_tokens=self.settings.llm_max_tokens
            )
        elif self.settings.llm_provider == LLMProvider.ANTHROPIC:
            if not self.settings.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY required for Anthropic provider")
            return AnthropicClient(
                api_key=self.settings.anthropic_api_key,
                model=self.settings.llm_model,
                max_tokens=self.settings.llm_max_tokens
            )
        else:
            raise ValueError(f"Unknown LLM provider: {self.settings.llm_provider}")

    def _create_embedding_provider(self) -> EmbeddingProviderBase:
        """Create the appropriate embedding provider."""
        if self.settings.embedding_provider == EmbeddingProvider.OPENAI:
            if not self.settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY required for OpenAI embeddings")
            return OpenAIEmbeddings(
                api_key=self.settings.openai_api_key,
                model=self.settings.embedding_model
            )
        else:
            return SentenceTransformerEmbeddings(
                model_name=self.settings.embedding_model
            )

    async def chat(self, user_input: str) -> ChatResponse:
        """
        Process a user message and generate a response.

        This is the main entry point for chat interactions.

        Args:
            user_input: The user's message

        Returns:
            ChatResponse with the assistant's reply and metadata
        """
        if not self._initialized:
            await self.initialize()

        logger.info("Processing chat request")
        was_summarized = False
        long_term_context_used = False
        input_filtered = False

        # STEP 1: Filter input
        filter_result = self._input_filter.filter(user_input)
        if filter_result.is_suspicious:
            logger.warning(f"Suspicious input detected: {filter_result.detections}")
            input_filtered = True

            if filter_result.risk_score >= self.settings.injection_risk_threshold:
                logger.error("Input blocked due to high risk score")
                return ChatResponse(
                    content="I apologize, but I cannot process that request.",
                    input_tokens=0,
                    output_tokens=0,
                    was_summarized=False,
                    long_term_context_used=False,
                    input_filtered=True
                )

        clean_input = filter_result.filtered

        # STEP 2: Retrieve relevant long-term memory
        long_term_context = ""
        try:
            long_term_context = await self._long_term_memory.get_context_for_query(
                query=clean_input,
                max_tokens=500
            )
            if long_term_context:
                long_term_context_used = True
                logger.info("Retrieved relevant long-term context")
        except Exception as e:
            logger.warning(f"Failed to retrieve long-term memory: {e}")

        # STEP 3: Add user message to short-term memory
        self._short_term_memory.add_message("user", clean_input)

        # STEP 4: Check if summarization is needed
        if self._short_term_memory.needs_summarization():
            logger.info("Summarization threshold reached, summarizing...")
            callback = create_summarizer_callback(self._summarizer)
            was_summarized = await self._short_term_memory.summarize_if_needed(callback)
            if was_summarized:
                logger.info("Conversation summarized successfully")
                stats = self._short_term_memory.get_stats()
                logger.info(f"Memory stats after summarization: {stats}")

        # STEP 5: Build context for LLM
        context_messages = self._short_term_memory.get_context_messages()

        # Build system prompt with long-term context
        system_prompt = self.settings.system_prompt
        if long_term_context:
            system_prompt = f"{system_prompt}\n\n{long_term_context}"

        # STEP 6: Send to LLM
        logger.info("Sending request to LLM")
        response = await self._llm_client.chat(
            system_prompt=system_prompt,
            messages=context_messages,
            max_tokens=self.settings.llm_max_tokens,
            temperature=self.settings.llm_temperature
        )

        # Log token usage
        log_token_usage(
            logger=logger,
            operation="chat",
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            model=response.model
        )

        # STEP 7: Validate output
        validation = self._output_validator.validate(response.content)
        if not validation.is_valid:
            logger.warning(f"Output validation failed: {validation.issues}")
            content = self._output_validator.get_safe_content(response.content)
        else:
            content = response.content

        # STEP 8: Store assistant response in short-term memory
        self._short_term_memory.add_message("assistant", content)

        logger.info("Chat request completed successfully")

        return ChatResponse(
            content=content,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            was_summarized=was_summarized,
            long_term_context_used=long_term_context_used,
            input_filtered=input_filtered
        )

    async def save_memory(self) -> None:
        """Persist long-term memory to disk."""
        if self._long_term_memory:
            await self._long_term_memory.save(self.settings.long_term_memory_path)
            logger.info("Long-term memory saved")

    async def store_summary_to_long_term(self, force: bool = False) -> None:
        """Store current conversation summary to long-term memory."""
        if not self._short_term_memory.has_summary and not force:
            return

        # Get messages and create summary for storage
        messages = self._short_term_memory.get_context_messages()
        if not messages:
            return

        summary_data = await self._summarizer.summarize_for_storage(messages)

        await self._long_term_memory.store(
            content=summary_data["summary"],
            topics=summary_data["topics"]
        )
        logger.info("Conversation summary stored in long-term memory")

    def get_stats(self) -> dict:
        """Get current chatbot statistics."""
        stats = {
            "initialized": self._initialized,
            "llm_provider": self.settings.llm_provider.value,
        }

        if self._initialized:
            stats["short_term_memory"] = self._short_term_memory.get_stats()
            stats["long_term_memory"] = self._long_term_memory.get_stats()
            stats["llm_usage"] = self._llm_client.get_usage_stats()

        return stats

    async def reset_conversation(self) -> None:
        """Reset the current conversation (clear short-term memory)."""
        if self._short_term_memory:
            # Store summary before clearing
            await self.store_summary_to_long_term()
            self._short_term_memory.clear()
            logger.info("Conversation reset")


# CLI interface for testing
async def main():
    """Simple CLI for testing the chatbot."""
    print("Context-Aware Chatbot")
    print("=" * 40)
    print("Type 'quit' to exit, 'stats' for statistics")
    print()

    chatbot = ContextAwareChatbot()
    await chatbot.initialize()

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "quit":
                print("Saving memory and exiting...")
                await chatbot.store_summary_to_long_term(force=True)
                await chatbot.save_memory()
                break

            if user_input.lower() == "stats":
                stats = chatbot.get_stats()
                print(f"\nStats: {stats}\n")
                continue

            response = await chatbot.chat(user_input)
            print(f"\nAssistant: {response.content}")
            print(f"  [tokens: in={response.input_tokens}, out={response.output_tokens}]")
            if response.was_summarized:
                print("  [conversation was summarized]")
            if response.long_term_context_used:
                print("  [used long-term memory]")
            print()

        except KeyboardInterrupt:
            print("\nExiting...")
            await chatbot.save_memory()
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
