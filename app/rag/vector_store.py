"""
Vector store interface.

Design Decisions:
- Abstract interface: Swap between FAISS, Chroma, etc.
- Document-oriented: Stores text + metadata + vectors
- Async-ready: Non-blocking operations
- Persistence: Save/load for durability
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime
import json
import os


@dataclass
class Document:
    """A document stored in the vector database."""
    id: str
    content: str
    embedding: list[float]
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        """Create from dict."""
        return cls(
            id=data["id"],
            content=data["content"],
            embedding=data["embedding"],
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"])
        )


@dataclass
class SearchResult:
    """A search result with similarity score."""
    document: Document
    score: float  # Similarity score (higher = more similar)


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    async def add(self, document: Document) -> None:
        """Add a document to the store."""
        pass

    @abstractmethod
    async def add_batch(self, documents: list[Document]) -> None:
        """Add multiple documents."""
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        filter_metadata: Optional[dict] = None
    ) -> list[SearchResult]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query vector
            k: Number of results to return
            filter_metadata: Optional metadata filter

        Returns:
            List of SearchResult sorted by similarity
        """
        pass

    @abstractmethod
    async def delete(self, document_id: str) -> bool:
        """Delete a document by ID."""
        pass

    @abstractmethod
    async def save(self, path: str) -> None:
        """Persist the store to disk."""
        pass

    @abstractmethod
    async def load(self, path: str) -> None:
        """Load the store from disk."""
        pass


class FAISSVectorStore(VectorStore):
    """
    FAISS-based vector store for local development.

    Uses Facebook's FAISS library for efficient similarity search.
    Good for small to medium datasets (< 1M documents).
    """

    def __init__(self, dimension: int):
        """
        Initialize FAISS store.

        Args:
            dimension: Embedding dimension
        """
        self.dimension = dimension
        self._index = None
        self._documents: dict[str, Document] = {}
        self._id_to_idx: dict[str, int] = {}
        self._idx_to_id: dict[int, str] = {}

    def _ensure_index(self):
        """Lazy initialization of FAISS index."""
        if self._index is None:
            import faiss
            self._index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine after normalization)

    def _normalize(self, embedding: list[float]) -> Any:
        """Normalize embedding for cosine similarity."""
        import numpy as np
        import faiss
        arr = np.array(embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(arr)
        return arr

    async def add(self, document: Document) -> None:
        """Add a document to the store."""
        import faiss

        self._ensure_index()

        # Normalize embedding for cosine similarity
        normalized = self._normalize(document.embedding)

        # Add to index
        idx = self._index.ntotal
        self._index.add(normalized)

        # Store document and mappings
        self._documents[document.id] = document
        self._id_to_idx[document.id] = idx
        self._idx_to_id[idx] = document.id

    async def add_batch(self, documents: list[Document]) -> None:
        """Add multiple documents."""
        for doc in documents:
            await self.add(doc)

    async def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        filter_metadata: Optional[dict] = None
    ) -> list[SearchResult]:
        """Search for similar documents."""
        self._ensure_index()

        if self._index.ntotal == 0:
            return []

        # Normalize query
        normalized_query = self._normalize(query_embedding)

        # Search
        actual_k = min(k, self._index.ntotal)
        scores, indices = self._index.search(normalized_query, actual_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for not found
                continue

            doc_id = self._idx_to_id.get(int(idx))
            if doc_id is None:
                continue

            doc = self._documents.get(doc_id)
            if doc is None:
                continue

            # Apply metadata filter if provided
            if filter_metadata:
                if not all(doc.metadata.get(k) == v for k, v in filter_metadata.items()):
                    continue

            results.append(SearchResult(document=doc, score=float(score)))

        return results

    async def delete(self, document_id: str) -> bool:
        """Delete a document by ID."""
        if document_id not in self._documents:
            return False

        # Note: FAISS doesn't support deletion directly
        # We just remove from our tracking - a rebuild would be needed for true deletion
        del self._documents[document_id]
        return True

    async def save(self, path: str) -> None:
        """Persist the store to disk."""
        import faiss

        os.makedirs(path, exist_ok=True)

        # Save FAISS index
        if self._index is not None:
            faiss.write_index(self._index, os.path.join(path, "index.faiss"))

        # Save documents and mappings
        data = {
            "dimension": self.dimension,
            "documents": {k: v.to_dict() for k, v in self._documents.items()},
            "id_to_idx": self._id_to_idx,
            "idx_to_id": {str(k): v for k, v in self._idx_to_id.items()}
        }
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(data, f)

    async def load(self, path: str) -> None:
        """Load the store from disk."""
        import faiss

        index_path = os.path.join(path, "index.faiss")
        metadata_path = os.path.join(path, "metadata.json")

        if os.path.exists(index_path):
            self._index = faiss.read_index(index_path)

        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                data = json.load(f)

            self.dimension = data["dimension"]
            self._documents = {
                k: Document.from_dict(v)
                for k, v in data["documents"].items()
            }
            self._id_to_idx = data["id_to_idx"]
            self._idx_to_id = {int(k): v for k, v in data["idx_to_id"].items()}
