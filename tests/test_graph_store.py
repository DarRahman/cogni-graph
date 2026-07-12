# tests/test_graph_store.py - Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation and hierarchical entity-relation extraction.
# Contributed by Claude Code

"""Unit tests for the NetworkX graph store."""

import os
import tempfile
from cognigraph.graph_store import NetworkXGraphStore
from cognigraph.models import Entity, Relationship


def test_add_and_get_entity() -> None:
    """Tests adding and retrieving an entity from the graph store."""
    store = NetworkXGraphStore()
    entity = Entity(
        id="alice",
        name="Alice",
        type="Person",
        description="A software engineer"
    )

    store.add_entity(entity)
    retrieved = store.get_entity("alice")

    assert retrieved is not None
    assert retrieved.id == "alice"
    assert retrieved.name == "Alice"
    assert retrieved.type == "Person"
    assert retrieved.description == "A software engineer"


def test_add_and_get_relationship() -> None:
    """Tests adding and retrieving relationships from the graph store."""
    store = NetworkXGraphStore()
    rel = Relationship(
        source="alice",
        target="google",
        type="WORKS_AT",
        description="Alice works at Google",
        weight=1.0
    )

    store.add_relationship(rel)
    relationships = store.get_relationships("alice")

    assert len(relationships) == 1
    assert relationships[0].source == "alice"
    assert relationships[0].target == "google"
    assert relationships[0].type == "WORKS_AT"


def test_merge_entities() -> None:
    """Tests merging two entities and re-routing their edges."""
    store = NetworkXGraphStore()

    # Add entities
    store.add_entity(Entity(id="alice", name="Alice", type="Person"))
    store.add_entity(Entity(id="alice_smith", name="Alice Smith", type="Person"))
    store.add_entity(Entity(id="google", name="Google", type="Organization"))

    # Add relationships
    store.add_relationship(Relationship(source="alice", target="google", type="WORKS_AT"))
    store.add_relationship(Relationship(source="alice_smith", target="google", type="WORKS_AT"))

    # Merge alice_smith into alice
    merged = Entity(
        id="alice",
        name="Alice Smith",
        type="Person",
        description="Merged profile"
    )
    store.merge_entities("alice", "alice_smith", merged)

    # Verify alice_smith is removed
    assert store.get_entity("alice_smith") is None

    # Verify alice exists and has updated info
    alice = store.get_entity("alice")
    assert alice is not None
    assert alice.name == "Alice Smith"

    # Verify relationships are consolidated (weight should be updated/summed)
    rels = store.get_relationships("alice")
    assert len(rels) == 1
    assert rels[0].weight == 2.0


def test_save_and_load_disk() -> None:
    """Tests saving and loading the graph store to/from disk."""
    store = NetworkXGraphStore()
    store.add_entity(Entity(id="alice", name="Alice", type="Person"))

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        store.save_to_disk(tmp_path)

        new_store = NetworkXGraphStore()
        new_store.load_from_disk(tmp_path)

        entity = new_store.get_entity("alice")
        assert entity is not None
        assert entity.name == "Alice"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
