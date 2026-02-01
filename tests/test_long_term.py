import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime
from app.memory.long_term import LongTermMemoryManager, Memory
from app.rag.vector_store import Document, SearchResult

@pytest.fixture
def mock_embedding_provider():
    provider = AsyncMock()
    provider.embed.return_value = [0.1, 0.2, 0.3]
    return provider

@pytest.fixture
def mock_vector_store():
    store = AsyncMock()
    return store

@pytest.fixture
def memory_manager(mock_embedding_provider, mock_vector_store):
    return LongTermMemoryManager(
        embedding_provider=mock_embedding_provider,
        vector_store=mock_vector_store,
        session_id="test-session"
    )

@pytest.mark.asyncio
async def test_store_memory(memory_manager, mock_embedding_provider, mock_vector_store):
    # Act
    doc_id = await memory_manager.store(
        content="test content",
        topics=["test"],
        metadata={"source": "user"}
    )

    # Assert
    assert doc_id is not None
    mock_embedding_provider.embed.assert_called_once_with("test content")
    mock_vector_store.add.assert_called_once()
    
    # Verify document structure
    call_args = mock_vector_store.add.call_args
    doc = call_args[0][0]
    assert isinstance(doc, Document)
    assert doc.content == "test content"
    assert doc.embedding == [0.1, 0.2, 0.3]
    assert doc.metadata["topics"] == ["test"]
    assert doc.metadata["session_id"] == "test-session"
    assert doc.metadata["source"] == "user"

@pytest.mark.asyncio
async def test_retrieve_memory(memory_manager, mock_embedding_provider, mock_vector_store):
    # Setup mock return values
    mock_doc = Document(
        id="doc1",
        content="retrieved content",
        embedding=[0.1, 0.2, 0.3],
        metadata={"topics": ["test"], "session_id": "other-session"}
    )
    mock_result = SearchResult(document=mock_doc, score=0.9)
    mock_vector_store.search.return_value = [mock_result]
    
    # Act
    memories = await memory_manager.retrieve("query")
    
    # Assert
    assert len(memories) == 1
    assert memories[0].content == "retrieved content"
    assert memories[0].relevance_score == 0.9
    mock_embedding_provider.embed.assert_called_once_with("query")

@pytest.mark.asyncio
async def test_retrieve_filters_low_score(memory_manager, mock_vector_store):
    # Setup low score result
    mock_doc = Document(
        id="doc1",
        content="low score content",
        embedding=[0.1, 0.2, 0.3],
        metadata={"session_id": "other-session"}
    )
    mock_result = SearchResult(document=mock_doc, score=0.4) # Below default 0.5
    mock_vector_store.search.return_value = [mock_result]
    
    # Act
    memories = await memory_manager.retrieve("query", min_score=0.5)
    
    # Assert
    assert len(memories) == 0

@pytest.mark.asyncio
async def test_retrieve_excludes_current_session(memory_manager, mock_vector_store):
    # Setup current session result
    mock_doc = Document(
        id="doc1",
        content="current session content",
        embedding=[0.1, 0.2, 0.3],
        metadata={"session_id": "test-session"} # Matches manager's session
    )
    mock_result = SearchResult(document=mock_doc, score=0.9)
    mock_vector_store.search.return_value = [mock_result]
    
    # Act
    memories = await memory_manager.retrieve("query", include_current_session=False)
    
    # Assert
    assert len(memories) == 0

@pytest.mark.asyncio
async def test_get_context_for_query(memory_manager, mock_vector_store):
    # Setup results
    mock_doc = Document(
        id="doc1",
        content="past context",
        embedding=[0.1, 0.2, 0.3],
        metadata={"topics": ["testing"], "session_id": "other-session"}
    )
    mock_result = SearchResult(document=mock_doc, score=0.9)
    mock_vector_store.search.return_value = [mock_result]
    
    # Act
    context = await memory_manager.get_context_for_query("query")
    
    # Assert
    assert "Relevant context from previous conversations:" in context
    assert "[testing]: past context" in context
