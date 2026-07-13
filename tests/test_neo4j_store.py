# tests/test_neo4j_store.py - Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation and hierarchical entity-relation extraction.
# Contributed by Claude Code

"""Unit tests for the Neo4j graph store using mocks."""

import sys
from unittest.mock import MagicMock, patch

# Mock neo4j module if not present
try:
    import neo4j
except ImportError:
    mock_neo4j = MagicMock()
    sys.modules['neo4j'] = mock_neo4j

from datetime import datetime
import pytest
from cognigraph.neo4j_store import Neo4jGraphStore, sanitize_rel_type
from cognigraph.models import Entity, Relationship


def test_sanitize_rel_type() -> None:
    """Tests relationship type sanitization."""
    assert sanitize_rel_type("works_at") == "WORKS_AT"
    assert sanitize_rel_type("KNOWS!") == "KNOWS"
    assert sanitize_rel_type("  ") == "RELATED_TO"


@patch("cognigraph.neo4j_store.NEO4J_AVAILABLE", True)
@patch("neo4j.GraphDatabase")
def test_neo4j_store_lifecycle(mock_graph_database: MagicMock) -> None:
    """Tests initialization and closing of Neo4jGraphStore."""
    mock_driver = MagicMock()
    mock_graph_database.driver.return_value = mock_driver

    store = Neo4jGraphStore(uri="bolt://localhost:7687", user="neo4j", password="password")
    assert store.driver == mock_driver

    store.close()
    mock_driver.close.assert_called_once()


@patch("cognigraph.neo4j_store.NEO4J_AVAILABLE", True)
@patch("neo4j.GraphDatabase")
def test_neo4j_store_add_entity(mock_graph_database: MagicMock) -> None:
    """Tests adding an entity to Neo4jGraphStore."""
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_graph_database.driver.return_value = mock_driver
    mock_driver.session.return_value.__enter__.return_value = mock_session

    store = Neo4jGraphStore(uri="bolt://localhost:7687", user="neo4j", password="password")
    entity = Entity(id="alice", name="Alice", type="Person", description="A developer")
    store.add_entity(entity)

    mock_session.run.assert_called_once()
    args, kwargs = mock_session.run.call_args
    assert "MERGE (e:Entity {id: $id})" in args[0]
    assert kwargs["id"] == "alice"
    assert kwargs["name"] == "Alice"
    assert kwargs["type"] == "Person"


@patch("cognigraph.neo4j_store.NEO4J_AVAILABLE", True)
@patch("neo4j.GraphDatabase")
def test_neo4j_store_add_relationship(mock_graph_database: MagicMock) -> None:
    """Tests adding a relationship to Neo4jGraphStore."""
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_graph_database.driver.return_value = mock_driver
    mock_driver.session.return_value.__enter__.return_value = mock_session

    store = Neo4jGraphStore(uri="bolt://localhost:7687", user="neo4j", password="password")
    rel = Relationship(source="alice", target="google", type="WORKS_AT", weight=1.5)
    store.add_relationship(rel)

    mock_session.run.assert_called_once()
    args, kwargs = mock_session.run.call_args
    assert "MERGE (s:Entity {id: $source})" in args[0]
    assert "MERGE (s)-[r:WORKS_AT]->(t)" in args[0]
    assert kwargs["source"] == "alice"
    assert kwargs["target"] == "google"
    assert kwargs["weight"] == 1.5


@patch("cognigraph.neo4j_store.NEO4J_AVAILABLE", True)
@patch("neo4j.GraphDatabase")
def test_neo4j_store_merge_entities(mock_graph_database: MagicMock) -> None:
    """Tests merging entities in Neo4jGraphStore."""
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_graph_database.driver.return_value = mock_driver
    mock_driver.session.return_value.__enter__.return_value = mock_session

    # Mock the results of fetching relationships for re-routing
    mock_result_out = MagicMock()
    mock_result_out.__iter__.return_value = [
        {
            "type": "WORKS_AT",
            "target": "google",
            "weight": 1.0,
            "description": "Works at Google",
            "properties_json": "{}",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
    ]
    mock_result_in = MagicMock()
    mock_result_in.__iter__.return_value = []
    
    mock_session.run.side_effect = [None, mock_result_out, None, mock_result_in, None]

    store = Neo4jGraphStore(uri="bolt://localhost:7687", user="neo4j", password="password")
    merged = Entity(id="alice", name="Alice Smith", type="Person")
    store.merge_entities("alice", "alice_smith", merged)

    # Verify that multiple queries were run (add entity, fetch out, create out, fetch in, delete old)
    assert mock_session.run.call_count >= 4
