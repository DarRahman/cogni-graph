# tests/test_visualization.py - Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation and hierarchical entity-relation extraction.
# Contributed by Claude Code

"""Unit tests for the visualization module."""

from cognigraph.graph_store import NetworkXGraphStore
from cognigraph.models import Entity, Relationship
from cognigraph.visualization import generate_visual_html


def test_generate_visual_html() -> None:
    """Tests that generate_visual_html returns a valid HTML string containing graph data."""
    store = NetworkXGraphStore()
    
    # Add some entities and relationships
    store.add_entity(Entity(id="alice", name="Alice", type="Person", description="A developer"))
    store.add_entity(Entity(id="google", name="Google", type="Organization", description="A company"))
    store.add_relationship(Relationship(source="alice", target="google", type="WORKS_AT", weight=1.0))
    
    html = generate_visual_html(store)
    
    # Verify HTML structure and content
    assert "<!DOCTYPE html>" in html
    assert "vis-network" in html
    assert "alice" in html
    assert "google" in html
    assert "WORKS_AT" in html
    assert "A developer" in html
