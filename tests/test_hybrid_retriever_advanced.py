# tests/test_hybrid_retriever_advanced.py - Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation and hierarchical entity-relation extraction.
# Contributed by Claude Code

"""Unit tests for the advanced hybrid retriever features."""

import pytest
from cognigraph.graph_store import NetworkXGraphStore

from cognigraph.vector_store import SimpleVectorStore
from cognigraph.retriever import HybridRetriever
from cognigraph.models import Entity, Relationship


@pytest.fixture
def populated_stores() -> tuple[NetworkXGraphStore, SimpleVectorStore]:
    """Creates and populates a graph store and vector store for testing."""
    graph_store = NetworkXGraphStore()
    vector_store = SimpleVectorStore()

    # Add entities
    entities = [
        Entity(id="alice", name="Alice", type="Person", description="A software engineer at Google"),
        Entity(id="bob", name="Bob", type="Person", description="A product manager at Google"),
        Entity(id="google", name="Google", type="Organization", description="A technology company"),
        Entity(id="python", name="Python", type="Technology", description="A programming language"),
        Entity(id="rust", name="Rust", type="Technology", description="A systems programming language"),
    ]
    for ent in entities:
        graph_store.add_entity(ent)

    # Add relationships
    relationships = [
        Relationship(source="alice", target="google", type="WORKS_AT", weight=1.0),
        Relationship(source="bob", target="google", type="WORKS_AT", weight=0.8),
        Relationship(source="alice", target="bob", type="KNOWS", weight=0.5),
        Relationship(source="alice", target="python", type="USES", weight=0.9),
        Relationship(source="bob", target="rust", type="USES", weight=0.4),
    ]
    for rel in relationships:
        graph_store.add_relationship(rel)

    # Add vectors (mock embeddings)
    vector_store.add_vector("alice", [1.0, 0.0, 0.0], {"name": "Alice"})
    vector_store.add_vector("bob", [0.8, 0.2, 0.0], {"name": "Bob"})
    vector_store.add_vector("google", [0.0, 1.0, 0.0], {"name": "Google"})
    vector_store.add_vector("python", [0.0, 0.0, 1.0], {"name": "Python"})
    vector_store.add_vector("rust", [0.1, 0.0, 0.9], {"name": "Rust"})

    return graph_store, vector_store


def test_hybrid_retrieval_rrf(populated_stores: tuple[NetworkXGraphStore, SimpleVectorStore]) -> None:
    """Tests hybrid retrieval using Reciprocal Rank Fusion (RRF)."""
    graph_store, vector_store = populated_stores
    retriever = HybridRetriever(graph_store, vector_store, use_rrf=True)

    # Query vector close to 'alice'
    query_vector = [0.95, 0.05, 0.0]
    result = retriever.retrieve(
        query="Who is Alice?",
        query_vector=query_vector,
        k=2,
        depth=1,
        max_nodes=3
    )

    # Verify results
    assert len(result.entities) <= 3
    entity_ids = {e.id for e in result.entities}
    assert "alice" in entity_ids
    # 'google' and 'bob' should be retrieved via graph traversal from 'alice'
    assert "google" in entity_ids or "bob" in entity_ids

    # Verify relationships connect the retrieved entities
    for rel in result.relationships:
        assert rel.source in entity_ids
        assert rel.target in entity_ids


def test_hybrid_retrieval_linear(populated_stores: tuple[NetworkXGraphStore, SimpleVectorStore]) -> None:
    """Tests hybrid retrieval using weighted linear combination."""
    graph_store, vector_store = populated_stores
    retriever = HybridRetriever(graph_store, vector_store, use_rrf=False, beta=0.7)

    # Query vector close to 'python'
    query_vector = [0.0, 0.0, 0.95]
    result = retriever.retrieve(
        query="Tell me about Python",
        query_vector=query_vector,
        k=1,
        depth=2,
        max_nodes=4
    )

    entity_ids = {e.id for e in result.entities}
    assert "python" in entity_ids
    # 'alice' uses python, so she should be traversed
    assert "alice" in entity_ids


def test_hybrid_retrieval_type_filtering(populated_stores: tuple[NetworkXGraphStore, SimpleVectorStore]) -> None:
    """Tests hybrid retrieval with entity type filtering."""
    graph_store, vector_store = populated_stores
    retriever = HybridRetriever(graph_store, vector_store)

    query_vector = [1.0, 0.0, 0.0]
    
    # Include only 'Person'
    result_include = retriever.retrieve(
        query="Alice",
        query_vector=query_vector,
        k=3,
        depth=2,
        include_types=["Person"]
    )
    for ent in result_include.entities:
        assert ent.type == "Person"

    # Exclude 'Organization'
    result_exclude = retriever.retrieve(
        query="Alice",
        query_vector=query_vector,
        k=3,
        depth=2,
        exclude_types=["Organization"]
    )
    for ent in result_exclude.entities:
        assert ent.type != "Organization"
