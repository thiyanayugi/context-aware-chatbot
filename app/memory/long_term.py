"""
Long-term memory using vector database.

Design Decisions:
- Semantic retrieval: Finds relevant past context using embeddings
- Summary-focused: Stores compressed summaries, not raw messages
- Session-aware: Tracks which session each memory belongs to
- Relevance scoring: Returns memories ranked by similarity
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import uuid

from app.rag.embeddings import EmbeddingProvider
from app.rag.vector_store import VectorStore, Document, SearchResult


@dataclass
class Memory:
    """A long-term memory entry."""
    id: str
    content: str
    topics: list[str]
    session_id: str
    created_at: datetime
    relevance_score: Optional[float] = None

    @classmethod
    def from_search_result(cls, result: SearchResult) -> "Memory":
        """Create Memory from a vector search result."""
        doc = result.document
        return cls(
            id=doc.id,
            content=doc.content,
            topics=doc.metadata.get("topics", []),
            session_id=doc.metadata.get("session_id", "unknown"),
            created_at=doc.created_at,
            relevance_score=result.score
        )


class LongTermMemoryManager:
    """
    Manages long-term memory storage and retrieval.

    Stores conversation summaries in a vector database for
    semantic retrieval of relevant past context.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        min_relevance_score: float = 0.5,
        session_id: Optional[str] = None
    ):
        """
        Initialize long-term memory.

        Args:
            embedding_provider: Provider for generating embeddings
            vector_store: Vector database for storage
            session_id: Current session identifier
        """
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.min_relevance_score = min_relevance_score
        self.session_id = session_id or str(uuid.uuid4())

    async def store(
        self,
        content: str,
        topics: Optional[list[str]] = None,
        metadata: Optional[dict] = None
    ) -> str:
        """
        Store a memory (typically a conversation summary).

        Args:
            content: The memory content to store
            topics: Optional list of topic keywords
            metadata: Optional additional metadata

        Returns:
            ID of the stored memory
        """
        # Generate embedding for the content
        embedding = await self.embedding_provider.embed(content)

        # Create document
        doc_id = str(uuid.uuid4())
        doc_metadata = {
            "session_id": self.session_id,
            "topics": topics or [],
            **(metadata or {})
        }

        document = Document(
            id=doc_id,
            content=content,
            embedding=embedding,
            metadata=doc_metadata
        )

        # Store in vector database
        await self.vector_store.add(document)

        return doc_id

    async def retrieve(
        self,
        query: str,
        k: int = 3,
        min_score: Optional[float] = None,
        include_current_session: bool = True
    ) -> list[Memory]:
        """
        Retrieve relevant memories for a query.

        Args:
            query: The query text (typically the user's current message)
            k: Maximum number of memories to return
            min_score: Minimum similarity score (0-1)
            include_current_session: Whether to include memories from current session

        Returns:
            List of relevant Memory objects, ranked by relevance
        """
        # Generate query embedding
        query_embedding = await self.embedding_provider.embed(query)

        # Search vector store
        results = await self.vector_store.search(
            query_embedding=query_embedding,
            k=k * 2  # Fetch extra to allow for filtering
        )

        # Convert to Memory objects and filter
        memories = []
        for result in results:
            # Skip low-relevance results
            threshold = min_score if min_score is not None else self.min_relevance_score
            if result.score < threshold:
                continue

            # Skip current session if not wanted
            if not include_current_session:
                if result.document.metadata.get("session_id") == self.session_id:
                    continue

            memories.append(Memory.from_search_result(result))

            if len(memories) >= k:
                break

        return memories

    async def retrieve_by_topics(
        self,
        topics: list[str],
        k: int = 3
    ) -> list[Memory]:
        """
        Retrieve memories matching specific topics.

        Args:
            topics: List of topic keywords
            k: Maximum number of memories to return

        Returns:
            List of matching Memory objects
        """
        # Create a query from topics
        query = " ".join(topics)
        return await self.retrieve(query, k=k)

    async def get_context_for_query(
        self,
        query: str,
        max_tokens: int = 500
    ) -> str:
        """
        Get formatted context string for LLM prompt.

        Args:
            query: The user's query
            max_tokens: Approximate max tokens for context

        Returns:
            Formatted context string
        """
        memories = await self.retrieve(query, k=3)

        if not memories:
            return ""

        # Format memories into context
        context_parts = ["Relevant context from previous conversations:"]

        for memory in memories:
            topic_str = ", ".join(memory.topics) if memory.topics else "general"
            context_parts.append(f"- [{topic_str}]: {memory.content}")

        return "\n".join(context_parts)

    async def save(self, path: str) -> None:
        """Persist memory to disk."""
        await self.vector_store.save(path)

    async def load(self, path: str) -> None:
        """Load memory from disk."""
        await self.vector_store.load(path)

    def get_stats(self) -> dict:
        """Get memory statistics."""
        return {
            "session_id": self.session_id,
            "embedding_dimension": self.embedding_provider.dimension
        }
