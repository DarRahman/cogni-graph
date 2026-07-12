# tests/test_vector_store.py - Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation and hierarchical entity-relation extraction.
# Contributed by Claude Code

"""Unit tests for the simple vector store."""

import os
import tempfile
from cognigraph.vector_store import SimpleVectorStore


def test_add_and_search_vector() -> None:
    """Tests adding vectors and performing similarity search."""
    store = SimpleVectorStore()

    # Add vectors
    store.add_vector("node1", [1.0, 0.0, 0.0], {"name": "Node 1"})
    store.add_vector("node2", [0.0, 1.0, 0.0], {"name": "Node 2"})
    store.add_vector("node3", [0.707, 0.707, 0.0], {"name": "Node 3"})

    # Search with query vector close to node1
    results = store.similarity_search([0.9, 0.1, 0.0], k=2)

    assert len(results) == 2
    assert results[0][0] == "node1"
    assert results[0][1] > 0.9
    assert results[1][0] == "node3"


def test_save_and_load_vector_store() -> None:
    """Tests saving and loading the vector store to/from disk."""
    store = SimpleVectorStore()
    store.add_vector("node1", [1.0, 2.0, 3.0], {"name": "Node 1"})

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        store.save_to_disk(tmp_path)

        new_store = SimpleVectorStore()
        new_store.load_from_disk(tmp_path)

        assert "node1" in new_store.vectors
        assert new_store.vectors["node1"] == [1.0, 2.0, 3.0]
        assert new_store.metadata["node1"] == {"name": "Node 1"}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
