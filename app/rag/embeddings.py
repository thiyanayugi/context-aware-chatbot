"""
Embedding generation for text.

Design Decisions:
- Provider-agnostic: Abstract base class with concrete implementations
- Async-ready: All methods are async for non-blocking I/O
- Batching support: Efficient bulk embedding generation
- Caching-ready: Interface supports caching implementations
"""

from abc import ABC, abstractmethod
from typing import Optional
import hashlib


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of embeddings."""
        pass


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI embedding provider using text-embedding-ada-002 or text-embedding-3-small."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimension: int = 1536
    ):
        """
        Initialize OpenAI embeddings.

        Args:
            api_key: OpenAI API key
            model: Model name (default: text-embedding-3-small)
            dimension: Embedding dimension (default: 1536)
        """
        self.api_key = api_key
        self.model = model
        self._dimension = dimension
        self._client: Optional[object] = None

    async def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        client = await self._get_client()
        response = await client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        client = await self._get_client()
        response = await client.embeddings.create(
            model=self.model,
            input=texts
        )
        # Sort by index to maintain order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

    @property
    def dimension(self) -> int:
        return self._dimension


class SentenceTransformerEmbeddings(EmbeddingProvider):
    """
    Local embedding provider using sentence-transformers.

    Good for development/testing without API costs.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize sentence transformer embeddings.

        Args:
            model_name: HuggingFace model name
        """
        self.model_name = model_name
        self._model = None
        self._dimension: Optional[int] = None

    def _get_model(self):
        """Lazy load the model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        model = self._get_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._get_model()  # Load model to get dimension
        return self._dimension


def text_hash(text: str) -> str:
    """Generate a hash for text (useful for caching embeddings)."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]
