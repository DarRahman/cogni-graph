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
