# tests/test_consolidation.py - Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation and hierarchical entity-relation extraction.
# Contributed by Claude Code

"""Unit tests for the consolidation pipeline algorithms."""

from datetime import datetime, timedelta
import pytest
from cognigraph.models import Entity, Relationship, ChatMessage
from cognigraph.graph_store import NetworkXGraphStore
from cognigraph.vector_store import SimpleVectorStore
from cognigraph.extractor import RuleBasedExtractor
from cognigraph.pipeline import (
    ConsolidationPipeline,
    MockEmbedder,
    jaro_winkler_similarity,
    cosine_similarity,
)


def test_jaro_winkler_similarity() -> None:
    """Tests Jaro-Winkler similarity calculation."""
    assert jaro_winkler_similarity("martha", "marhta") > 0.9
    assert jaro_winkler_similarity("dwayne", "duane") > 0.8
    assert jaro_winkler_similarity("DIXON", "dicksonx") > 0.7
    assert jaro_winkler_similarity("apple", "microsoft") < 0.5
    assert jaro_winkler_similarity("", "") == 0.0


def test_cosine_similarity() -> None:
    """Tests cosine similarity calculation."""
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine_similarity([1.0, 1.0], [-1.0, -1.0]) == pytest.approx(-1.0)


def test_type_compatibility() -> None:
    """Tests type compatibility logic in pipeline."""
    pipeline = ConsolidationPipeline(
        NetworkXGraphStore(),
        SimpleVectorStore(),
        RuleBasedExtractor(),
        MockEmbedder()
    )
    assert pipeline._are_types_compatible("Person", "Person")
    assert pipeline._are_types_compatible("Person", "Concept")
    assert pipeline._are_types_compatible("Concept", "Organization")
    assert not pipeline._are_types_compatible("Person", "Organization")


def test_consolidation_decay_and_pruning() -> None:
    """Tests relationship weight decay and pruning during consolidation."""
    graph_store = NetworkXGraphStore()
    vector_store = SimpleVectorStore()
    extractor = RuleBasedExtractor()
    embedder = MockEmbedder(dimension=64)
    pipeline = ConsolidationPipeline(graph_store, vector_store, extractor, embedder)

    # Add entities and relationships
    now = datetime.utcnow()
    graph_store.add_entity(Entity(id="alice", name="Alice", type="Person"))
    graph_store.add_entity(Entity(id="google", name="Google", type="Organization"))
    
    # Add a relationship that is old (should decay and prune)
    old_rel = Relationship(
        source="alice",
        target="google",
        type="WORKS_AT",
        weight=0.15,
        created_at=now - timedelta(days=10),
        updated_at=now - timedelta(days=10)
    )
    graph_store.add_relationship(old_rel)

    # Run consolidation
    pipeline.consolidate()

    # Verify relationship is pruned (since weight decays below 0.1 threshold)
    rels = graph_store.get_relationships("alice")
    assert len(rels) == 0

    # Verify isolated nodes are pruned
    assert graph_store.get_entity("alice") is None
    assert graph_store.get_entity("google") is None


def test_consolidation_entity_merging() -> None:
    """Tests duplicate resolution and entity merging during consolidation."""
    graph_store = NetworkXGraphStore()
    vector_store = SimpleVectorStore()
    extractor = RuleBasedExtractor()
    embedder = MockEmbedder(dimension=64)
    pipeline = ConsolidationPipeline(graph_store, vector_store, extractor, embedder)

    # Add duplicate entities
    ent1 = Entity(id="alice", name="Alice", type="Person", description="Software engineer")
    ent2 = Entity(id="alice_smith", name="Alice Smith", type="Concept", description="Works at Google")
    
    graph_store.add_entity(ent1)
    graph_store.add_entity(ent2)

    # Add vectors (use the same vector to ensure high similarity)
    vec = embedder.embed_text("Alice: Software engineer")
    vector_store.add_vector("alice", vec, {})
    vector_store.add_vector("alice_smith", vec, {})

    # Add relationships
    graph_store.add_relationship(Relationship(source="alice", target="python", type="LIKES"))
    graph_store.add_relationship(Relationship(source="alice_smith", target="google", type="WORKS_AT"))

    # Run consolidation
    pipeline.consolidate()

    # Verify alice_smith is merged into alice
    assert graph_store.get_entity("alice_smith") is None
    
    primary = graph_store.get_entity("alice")
    assert primary is not None
    assert primary.type == "Person"  # Type resolved from Concept to Person
    assert "Software engineer" in primary.description
    assert "Works at Google" in primary.description

    # Verify relationships are re-routed to primary
    rels = graph_store.get_relationships("alice")
    rel_targets = {r.target for r in rels}
    assert "python" in rel_targets
    assert "google" in rel_targets

    # Verify vector store is updated
    assert "alice_smith" not in vector_store.vectors
    assert "alice" in vector_store.vectors
