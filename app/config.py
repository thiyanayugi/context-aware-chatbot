"""
Configuration settings for the chatbot application.

Design Decisions:
- Pydantic-based: Type-safe configuration with validation
- Environment-aware: Loads from env vars with sensible defaults
- Centralized: All settings in one place
- Immutable: Settings don't change at runtime
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from enum import Enum


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""
    OPENAI = "openai"
    SENTENCE_TRANSFORMER = "sentence_transformer"


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # LLM Configuration
    llm_provider: LLMProvider = Field(default=LLMProvider.OPENAI)
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    llm_model: str = Field(default="gpt-4o-mini")
    llm_max_tokens: int = Field(default=1000)
    llm_temperature: float = Field(default=0.7)

    # Context Window Configuration
    max_context_tokens: int = Field(default=8000)
    summarization_threshold: float = Field(default=0.7)
    preserve_recent_messages: int = Field(default=4)
    response_buffer_tokens: int = Field(default=1000)

    # Embedding Configuration
    embedding_provider: EmbeddingProvider = Field(default=EmbeddingProvider.SENTENCE_TRANSFORMER)
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    embedding_dimension: int = Field(default=384)

    # Memory Configuration
    long_term_memory_path: str = Field(default="./data/memory")
    min_relevance_score: float = Field(default=0.5)
    max_retrieved_memories: int = Field(default=3)

    # Guardrails Configuration
    max_input_length: int = Field(default=10000)
    max_output_length: int = Field(default=10000)
    enable_injection_detection: bool = Field(default=True)
    injection_risk_threshold: float = Field(default=0.5)

    # Logging Configuration
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")  # "json" or "text"

    # System Prompt
    system_prompt: str = Field(
        default="""You are a helpful AI assistant. You have access to conversation history and relevant context from previous conversations. Use this context to provide accurate, personalized responses.

Guidelines:
- Be concise and helpful
- Reference relevant past context when applicable
- Ask for clarification if the request is unclear
- Admit when you don't know something"""
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the application settings."""
    return settings
