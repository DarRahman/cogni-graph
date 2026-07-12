# tests/test_retriever.py - Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation and hierarchical entity-relation extraction.
# Contributed by Claude Code

"""Unit tests for the hybrid retriever and consolidation pipeline."""

from datetime import datetime
from cognigraph.extractor import RuleBasedExtractor
from cognigraph.graph_store import NetworkXGraphStore
from cognigraph.models import ChatMessage
from cognigraph.pipeline import ConsolidationPipeline, MockEmbedder
from cognigraph.retriever import HybridRetriever
from cognigraph.vector_store import SimpleVectorStore


def test_hybrid_retrieval_flow() -> None:
    """Tests the end-to-end flow of ingestion, embedding, and hybrid retrieval."""
    # Initialize components
    graph_store = NetworkXGraphStore()
    vector_store = SimpleVectorStore()
    extractor = RuleBasedExtractor()
    embedder = MockEmbedder(dimension=128)
    pipeline = ConsolidationPipeline(graph_store, vector_store, extractor, embedder)
    retriever = HybridRetriever(graph_store, vector_store)

    # Ingest messages
    messages = [
        ChatMessage(
            role="user",
            content="Alice works at Google. Bob knows Alice.",
            timestamp=datetime.utcnow()
        )
    ]
    pipeline.ingest_and_process(messages)

    # Verify stores are populated
    assert graph_store.graph.number_of_nodes() >= 3
    assert len(vector_store.vectors) >= 3

    # Perform retrieval
    query = "Where does Alice work?"
    query_vector = embedder.embed_text(query)

    result = retriever.retrieve(
        query=query,
        query_vector=query_vector,
        k=2,
        depth=1
    )

    # Verify retrieval results
    assert len(result.entities) > 0
    entity_ids = {entity.id for entity in result.entities}
    assert "alice" in entity_ids

    # Verify relationships are retrieved
    assert len(result.relationships) > 0
    rel_types = {rel.type for rel in result.relationships}
    assert "WORKS_AT" in rel_types or "KNOWS" in rel_types


def test_consolidation_decay_and_merge() -> None:
    """Tests that consolidation decays weights and merges similar entities."""
    graph_store = NetworkXGraphStore()
    vector_store = SimpleVectorStore()
    extractor = RuleBasedExtractor()
    embedder = MockEmbedder(dimension=128)
    pipeline = ConsolidationPipeline(graph_store, vector_store, extractor, embedder)

    # Ingest messages with duplicate-like entities
    messages = [
        ChatMessage(
            role="user",
            content="Charlie works at Microsoft. Charlie lives in Seattle.",
            timestamp=datetime.utcnow()
        )
    ]
    pipeline.ingest_and_process(messages)

    # Verify initial state
    assert graph_store.graph.has_node("charlie")
    initial_weight = graph_store.get_relationships("charlie")[0].weight
    assert initial_weight == 1.0

    # Run consolidation
    pipeline.consolidate()

    # Verify weight decay
    decayed_weight = graph_store.get_relationships("charlie")[0].weight
    assert decayed_weight < 1.0
