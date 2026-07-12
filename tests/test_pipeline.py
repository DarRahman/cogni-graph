# tests/test_pipeline.py - Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation and hierarchical entity-relation extraction.
# Contributed by Claude Code

"""Unit tests for the consolidation pipeline and mock embedder."""

import numpy as np
from cognigraph.pipeline import MockEmbedder


def test_mock_embedder_determinism() -> None:
    """Tests that the mock embedder produces deterministic unit vectors."""
    embedder = MockEmbedder(dimension=64)

    vec1 = embedder.embed_text("hello world")
    vec2 = embedder.embed_text("hello world")
    vec3 = embedder.embed_text("different text")

    assert len(vec1) == 64
    assert vec1 == vec2
    assert vec1 != vec3

    # Check unit length
    norm = np.linalg.norm(vec1)
    assert np.isclose(norm, 1.0)


def test_pipeline_with_episodic_buffer() -> None:
    """Tests that the pipeline correctly interacts with the episodic buffer."""
    from datetime import datetime
    from cognigraph.extractor import RuleBasedExtractor
    from cognigraph.graph_store import NetworkXGraphStore
    from cognigraph.models import ChatMessage
    from cognigraph.pipeline import ConsolidationPipeline, MockEmbedder
    from cognigraph.vector_store import SimpleVectorStore
    from cognigraph.episodic_buffer import EpisodicBuffer

    graph_store = NetworkXGraphStore()
    vector_store = SimpleVectorStore()
    episodic_buffer = EpisodicBuffer()
    extractor = RuleBasedExtractor()
    embedder = MockEmbedder(dimension=64)
    pipeline = ConsolidationPipeline(graph_store, vector_store, extractor, embedder, episodic_buffer)

    # Ingest messages
    messages = [
        ChatMessage(
            role="user",
            content="Alice works at Google.",
            timestamp=datetime.utcnow()
        )
    ]
    pipeline.ingest_and_process(messages)

    # Verify messages are in episodic buffer and marked as processed
    stored_msgs = episodic_buffer.get_messages()
    assert len(stored_msgs) == 1
    assert stored_msgs[0].content == "Alice works at Google."
    assert stored_msgs[0].processed

    # Verify entities are in graph store
    assert graph_store.get_entity("alice") is not None

    # Now let's add a message directly to episodic buffer (unprocessed)
    msg2 = ChatMessage(
        role="user",
        content="Bob likes Python.",
        timestamp=datetime.utcnow()
    )
    episodic_buffer.add_message(msg2)

    # Verify it is unprocessed
    unprocessed = episodic_buffer.get_messages(unprocessed_only=True)
    assert len(unprocessed) == 1
    assert unprocessed[0].content == "Bob likes Python."

    # Run consolidation
    pipeline.consolidate()

    # Verify it is now processed
    assert len(episodic_buffer.get_messages(unprocessed_only=True)) == 0

    # Verify Bob and Python are in graph store
    assert graph_store.get_entity("bob") is not None
    assert graph_store.get_entity("python") is not None
