# tests/test_vector_store_advanced.py - Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation and hierarchical entity-relation extraction.
# Contributed by Claude Code

"""Unit tests for Chroma and Qdrant vector stores."""

import os
import tempfile
import pytest
from cognigraph.vector_store import ChromaVectorStore, QdrantVectorStore, CHROMA_AVAILABLE, QDRANT_AVAILABLE


@pytest.mark.skipif(not CHROMA_AVAILABLE, reason="chromadb package not installed")
def test_chroma_vector_store() -> None:
    """Tests basic operations of ChromaVectorStore."""
    # To prevent Windows PermissionError, we use a directory that is not deleted inside the test,
    # or handle cleanup gracefully, ignoring errors.
    import shutil
    tmpdir = tempfile.mkdtemp()
    try:
        store = ChromaVectorStore(path=tmpdir, collection_name="test_collection")
        
        # Add vectors
        store.add_vector("node1", [1.0, 0.0, 0.0], {"name": "Node 1", "tags": ["a", "b"]})
        store.add_vector("node2", [0.0, 1.0, 0.0], {"name": "Node 2"})
        
        assert store.count() == 2
        
        # Retrieve vector
        vec = store.get_vector("node1")
        assert vec is not None
        assert len(vec) == 3
        assert vec[0] == pytest.approx(1.0)
        
        # Search
        results = store.similarity_search([0.9, 0.1, 0.0], k=1)
        assert len(results) == 1
        assert results[0][0] == "node1"
        assert results[0][2]["name"] == "Node 1"
        
        # Delete
        store.delete_vector("node1")
        assert store.count() == 1
        assert store.get_vector("node1") is None
    finally:
        # Release references
        if 'store' in locals():
            try:
                store.client = None
                store.collection = None
            except Exception:
                pass
            del store
        import gc
        gc.collect()
        # Clean up directory, ignore errors on Windows if files are still locked
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client package not installed")
def test_qdrant_vector_store() -> None:
    """Tests basic operations of QdrantVectorStore."""
    store = QdrantVectorStore(url=":memory:", collection_name="test_collection", dimension=3)
    
    # Add vectors
    store.add_vector("node1", [1.0, 0.0, 0.0], {"name": "Node 1", "tags": ["a", "b"]})
    store.add_vector("node2", [0.0, 1.0, 0.0], {"name": "Node 2"})
    
    assert store.count() == 2
    
    # Retrieve vector
    vec = store.get_vector("node1")
    assert vec is not None
    assert len(vec) == 3
    assert vec[0] == pytest.approx(1.0)
    
    # Search
    results = store.similarity_search([0.9, 0.1, 0.0], k=1)
    assert len(results) == 1
    assert results[0][0] == "node1"
    assert results[0][2]["name"] == "Node 1"
    
    # Delete
    store.delete_vector("node1")
    assert store.count() == 1
    assert store.get_vector("node1") is None
