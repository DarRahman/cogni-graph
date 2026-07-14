# tests/test_consolidation_graph.py - Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation and hierarchical entity-relation extraction.
# Contributed by Claude Code

"""Unit tests for the LangGraph consolidation workflow."""

from datetime import datetime, timedelta
import pytest
from cognigraph.consolidation_graph import LangGraphConsolidator
from cognigraph.episodic_buffer import EpisodicBuffer
from cognigraph.extractor import RuleBasedExtractor
from cognigraph.graph_store import NetworkXGraphStore
from cognigraph.models import ChatMessage, Entity, Relationship
from cognigraph.pipeline import MockEmbedder
from cognigraph.vector_store import SimpleVectorStore


@pytest.fixture
def consolidator_setup() -> tuple[
    NetworkXGraphStore,
    SimpleVectorStore,
    RuleBasedExtractor,
    MockEmbedder,
    EpisodicBuffer,
    LangGraphConsolidator,
]:
    """Sets up components for testing the LangGraph consolidator."""
    graph_store = NetworkXGraphStore()
    vector_store = SimpleVectorStore()
    extractor = RuleBasedExtractor()
    embedder = MockEmbedder(dimension=64)
    episodic_buffer = EpisodicBuffer()
    consolidator = LangGraphConsolidator(
        graph_store=graph_store,
        vector_store=vector_store,
        extractor=extractor,
        embedder=embedder,
        episodic_buffer=episodic_buffer,
    )
    return graph_store, vector_store, extractor, embedder, episodic_buffer, consolidator


def test_workflow_with_new_messages(consolidator_setup: tuple[
    NetworkXGraphStore,
    SimpleVectorStore,
    RuleBasedExtractor,
    MockEmbedder,
    EpisodicBuffer,
    LangGraphConsolidator,
]) -> None:
    """Tests running the workflow with new messages in the episodic buffer."""
    graph_store, vector_store, _, _, episodic_buffer, consolidator = consolidator_setup

    # Add messages to episodic buffer
    msg = ChatMessage(
        role="user",
        content="Alice works at Google. Bob knows Alice.",
        timestamp=datetime.utcnow(),
        metadata={"session_id": "session_1"},
    )
    episodic_buffer.add_message(msg)

    # Run workflow
    result = consolidator.run_consolidation_workflow(session_id="session_1")

    # Verify final state
    assert result["status"] == "success"
    assert len(result["unprocessed_message_ids"]) == 1
    assert result["completed_at"] is not None

    # Verify entities and relationships are in graph store
    assert graph_store.get_entity("alice") is not None
    assert graph_store.get_entity("google") is not None
    assert graph_store.get_entity("bob") is not None

    # Verify relationships are created
    rels = graph_store.get_relationships("alice")
    assert len(rels) > 0
    assert any(r.target == "google" and r.type == "WORKS_AT" for r in rels)

    # Verify messages are marked as processed
    unprocessed = episodic_buffer.get_messages(session_id="session_1", unprocessed_only=True)
    assert len(unprocessed) == 0


def test_workflow_no_new_messages(consolidator_setup: tuple[
    NetworkXGraphStore,
    SimpleVectorStore,
    RuleBasedExtractor,
    MockEmbedder,
    EpisodicBuffer,
    LangGraphConsolidator,
]) -> None:
    """Tests running the workflow when there are no new messages."""
    graph_store, vector_store, _, _, _, consolidator = consolidator_setup

    # Add an existing entity and relationship to the graph
    now = datetime.utcnow()
    graph_store.add_entity(Entity(id="alice", name="Alice", type="Person", updated_at=now))
    graph_store.add_entity(Entity(id="google", name="Google", type="Organization", updated_at=now))
    graph_store.add_relationship(
        Relationship(
            source="alice",
            target="google",
            type="WORKS_AT",
            weight=1.0,
            updated_at=now - timedelta(days=2),
        )
    )

    # Run workflow (no messages in episodic buffer)
    result = consolidator.run_consolidation_workflow(decay_factor=0.9)

    # Verify final state
    assert result["status"] == "success"
    assert len(result["unprocessed_message_ids"]) == 0

    # Verify decay happened (weight should be decayed from 1.0)
    rels = graph_store.get_relationships("alice")
    assert len(rels) == 1
    assert rels[0].weight < 1.0


def test_workflow_forgetting_policy(consolidator_setup: tuple[
    NetworkXGraphStore,
    SimpleVectorStore,
    RuleBasedExtractor,
    MockEmbedder,
    EpisodicBuffer,
    LangGraphConsolidator,
]) -> None:
    """Tests the forgetting policy node in the workflow."""
    graph_store, vector_store, _, embedder, _, consolidator = consolidator_setup

    # Add an old entity (should be forgotten)
    old_time = datetime.utcnow() - timedelta(days=40)
    old_entity = Entity(
        id="old_concept",
        name="Old Concept",
        type="Concept",
        description="Something forgotten",
        created_at=old_time,
        updated_at=old_time,
    )
    graph_store.add_entity(old_entity)
    vector_store.add_vector("old_concept", embedder.embed_text("Old Concept"), {})

    # Add a permanent old entity (should NOT be forgotten)
    permanent_entity = Entity(
        id="permanent_concept",
        name="Permanent Concept",
        type="Concept",
        description="Something important",
        properties={"permanent": True},
        created_at=old_time,
        updated_at=old_time,
    )
    graph_store.add_entity(permanent_entity)
    vector_store.add_vector("permanent_concept", embedder.embed_text("Permanent Concept"), {})

    # Run workflow with forgetting threshold of 30 days
    result = consolidator.run_consolidation_workflow(forgetting_age_days=30.0)

    assert result["status"] == "success"
    assert result["forgotten_entities_count"] == 1

    # Verify old_concept is removed
    assert graph_store.get_entity("old_concept") is None
    assert vector_store.get_vector("old_concept") is None

    # Verify permanent_concept is retained
    assert graph_store.get_entity("permanent_concept") is not None
    assert vector_store.get_vector("permanent_concept") is not None


def test_workflow_entity_resolution(consolidator_setup: tuple[
    NetworkXGraphStore,
    SimpleVectorStore,
    RuleBasedExtractor,
    MockEmbedder,
    EpisodicBuffer,
    LangGraphConsolidator,
]) -> None:
    """Tests entity resolution and merging in the workflow."""
    graph_store, vector_store, _, embedder, _, consolidator = consolidator_setup

    # Add duplicate entities
    ent1 = Entity(id="charlie", name="Charlie", type="Person", description="A developer")
    ent2 = Entity(id="charlie_smith", name="Charlie Smith", type="Concept", description="A developer")
    graph_store.add_entity(ent1)
    graph_store.add_entity(ent2)

    # Add identical vectors to ensure high similarity
    vec = embedder.embed_text("Charlie: A developer")
    vector_store.add_vector("charlie", vec, {})
    vector_store.add_vector("charlie_smith", vec, {})

    # Add relationships
    graph_store.add_relationship(Relationship(source="charlie", target="python", type="USES"))
    graph_store.add_relationship(Relationship(source="charlie_smith", target="google", type="WORKS_AT"))

    # Run workflow
    result = consolidator.run_consolidation_workflow(similarity_threshold=0.8)

    assert result["status"] == "success"
    assert result["merged_entities_count"] == 1

    # Verify charlie_smith is merged into charlie
    assert graph_store.get_entity("charlie_smith") is None
    assert vector_store.get_vector("charlie_smith") is None

    primary = graph_store.get_entity("charlie")
    assert primary is not None
    assert primary.type == "Person"  # Type resolved from Concept to Person

    # Verify relationships are re-routed
    rels = graph_store.get_relationships("charlie")
    targets = {r.target for r in rels}
    assert "python" in targets
    assert "google" in targets
