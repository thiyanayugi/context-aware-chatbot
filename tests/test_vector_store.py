import pytest
from unittest.mock import Mock, patch, mock_open, MagicMock
from app.rag.vector_store import FAISSVectorStore, Document, SearchResult
import numpy as np

@pytest.fixture
def mock_faiss():
    with patch.dict('sys.modules', {'faiss': MagicMock()}):
        import faiss
        # Setup index mock
        index_mock = MagicMock()
        index_mock.ntotal = 0
        faiss.IndexFlatIP.return_value = index_mock
        yield faiss

@pytest.fixture
def vector_store(mock_faiss):
    return FAISSVectorStore(dimension=384)

@pytest.mark.asyncio
async def test_ensure_index(vector_store, mock_faiss):
    vector_store._ensure_index()
    mock_faiss.IndexFlatIP.assert_called_with(384)
    assert vector_store._index is not None

@pytest.mark.asyncio
async def test_add_document(vector_store, mock_faiss):
    # Setup
    doc = Document(id="1", content="test", embedding=[0.1] * 384)
    # Mock index.ntotal scaling
    vector_store._ensure_index()
    vector_store._index.ntotal = 0
    
    # Act
    await vector_store.add(doc)
    
    # Assert
    assert "1" in vector_store._documents
    assert vector_store._documents["1"] == doc
    assert vector_store._id_to_idx["1"] == 0
    vector_store._index.add.assert_called()

@pytest.mark.asyncio
async def test_search_documents(vector_store, mock_faiss):
    # Setup
    doc = Document(id="1", content="test", embedding=[0.1] * 384)
    vector_store._documents["1"] = doc
    vector_store._id_to_idx["1"] = 0
    vector_store._idx_to_id[0] = "1"
    
    vector_store._ensure_index()
    vector_store._index.ntotal = 1
    
    # Mock search return: scores, indices
    # FAISS returns (scores, indices) as numpy arrays
    vector_store._index.search.return_value = (
        np.array([[0.9]]), 
        np.array([[0]])
    )
    
    query = [0.1] * 384
    
    # Act
    results = await vector_store.search(query, k=1)
    
    # Assert
    assert len(results) == 1
    assert results[0].document.id == "1"
    assert results[0].score == 0.9

@pytest.mark.asyncio
async def test_search_with_metadata_filter(vector_store, mock_faiss):
    # Setup
    doc1 = Document(id="1", content="test1", embedding=[0.1] * 384, metadata={"type": "A"})
    doc2 = Document(id="2", content="test2", embedding=[0.1] * 384, metadata={"type": "B"})
    
    vector_store._documents = {"1": doc1, "2": doc2}
    vector_store._idx_to_id = {0: "1", 1: "2"}
    
    vector_store._ensure_index()
    vector_store._index.ntotal = 2
    
    # Mock search returning both
    vector_store._index.search.return_value = (
        np.array([[0.9, 0.8]]), 
        np.array([[0, 1]])
    )
    
    # Act
    results = await vector_store.search([0.1]*384, k=2, filter_metadata={"type": "B"})
    
    # Assert
    assert len(results) == 1
    assert results[0].document.id == "2"

@pytest.mark.asyncio
async def test_save_and_load(vector_store, mock_faiss):
    # Setup
    doc = Document(id="1", content="test", embedding=[0.1] * 384)
    await vector_store.add(doc)
    
    with patch("os.makedirs") as mock_makedirs, \
         patch("builtins.open", mock_open()) as mock_file, \
         patch("json.dump") as mock_json_dump:
        
        # Act
        await vector_store.save("test_path")
        
        # Assert
        mock_makedirs.assert_called_with("test_path", exist_ok=True)
        mock_faiss.write_index.assert_called()
        mock_json_dump.assert_called()

@pytest.mark.asyncio
async def test_delete(vector_store):
    doc = Document(id="1", content="test", embedding=[0.1] * 384)
    vector_store._documents["1"] = doc
    
    # Act
    success = await vector_store.delete("1")
    
    # Assert
    assert success is True
    assert "1" not in vector_store._documents
    
    # Try deleting non-existent
    success = await vector_store.delete("999")
    assert success is False
