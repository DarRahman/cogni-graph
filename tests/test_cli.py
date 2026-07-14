# tests/test_cli.py - Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation and hierarchical entity-relation extraction.
# Contributed by Claude Code

"""Unit tests for the CogniGraph CLI tool."""

import argparse
from unittest.mock import MagicMock, patch
import pytest

from cognigraph.cli import (
    cmd_status,
    cmd_query,
    cmd_consolidate,
    cmd_buffer,
    cmd_visualize,
    init_components,
)
from cognigraph.graph_store import NetworkXGraphStore
from cognigraph.vector_store import SimpleVectorStore
from cognigraph.episodic_buffer import EpisodicBuffer
from cognigraph.models import Entity, Relationship, ChatMessage, StoredMessage
from cognigraph.extractor import RuleBasedExtractor
from cognigraph.pipeline import MockEmbedder


@pytest.fixture
def mock_stores() -> tuple[MagicMock, MagicMock, MagicMock]:
    """Creates mocked stores for CLI testing."""
    graph_store = MagicMock(spec=NetworkXGraphStore)
    graph_store.graph = MagicMock()
    vector_store = MagicMock(spec=SimpleVectorStore)
    episodic_buffer = MagicMock(spec=EpisodicBuffer)
    return graph_store, vector_store, episodic_buffer


@patch("cognigraph.cli.load_stores")
def test_cmd_status(mock_load: MagicMock, mock_stores: tuple[MagicMock, MagicMock, MagicMock]) -> None:
    """Tests the status CLI command."""
    graph_store, vector_store, episodic_buffer = mock_stores
    
    # Setup mocks
    graph_store.get_all_entities.return_value = []
    graph_store.get_all_relationships.return_value = []
    vector_store.count.return_value = 5
    episodic_buffer.messages = {"1": MagicMock(), "2": MagicMock()}
    episodic_buffer.get_messages.return_value = []

    args = argparse.Namespace()
    cmd_status(args, graph_store, vector_store, episodic_buffer)

    mock_load.assert_called_once_with(graph_store, vector_store, episodic_buffer)
    vector_store.count.assert_called_once()


@patch("cognigraph.cli.load_stores")
@patch("cognigraph.cli.HybridRetriever")
def test_cmd_query(mock_retriever_cls: MagicMock, mock_load: MagicMock, mock_stores: tuple[MagicMock, MagicMock, MagicMock]) -> None:
    """Tests the query CLI command."""
    graph_store, vector_store, _ = mock_stores
    mock_retriever = MagicMock()
    mock_retriever_cls.return_value = mock_retriever
    
    mock_result = MagicMock()
    mock_result.entities = [Entity(id="alice", name="Alice", type="Person")]
    mock_result.relationships = []
    mock_result.scores = {"alice": 0.95}
    mock_retriever.retrieve.return_value = mock_result

    mock_embedder = MagicMock()
    mock_embedder.embed_text.return_value = [0.1] * 64

    args = argparse.Namespace(query="Who is Alice?", k=3, depth=2, max_nodes=10)
    cmd_query(args, graph_store, vector_store, mock_embedder)

    mock_embedder.embed_text.assert_called_once_with("Who is Alice?")
    mock_retriever.retrieve.assert_called_once()


@patch("cognigraph.cli.load_stores")
@patch("cognigraph.cli.save_stores")
@patch("cognigraph.cli.LangGraphConsolidator")
def test_cmd_consolidate(mock_consolidator_cls: MagicMock, mock_save: MagicMock, mock_load: MagicMock, mock_stores: tuple[MagicMock, MagicMock, MagicMock]) -> None:
    """Tests the consolidate CLI command."""
    graph_store, vector_store, episodic_buffer = mock_stores
    mock_consolidator = MagicMock()
    mock_consolidator_cls.return_value = mock_consolidator
    mock_consolidator.run_consolidation_workflow.return_value = {
        "status": "success",
        "unprocessed_message_ids": ["1"],
        "merged_entities_count": 1,
        "pruned_relationships_count": 0,
        "forgotten_entities_count": 0
    }

    args = argparse.Namespace(
        session_id="session_1",
        decay_factor=0.95,
        pruning_threshold=0.1,
        similarity_threshold=0.85,
        forgetting_age_days=30.0
    )
    cmd_consolidate(args, graph_store, vector_store, episodic_buffer, MagicMock(), MagicMock())

    mock_load.assert_called_once_with(graph_store, vector_store, episodic_buffer)
    mock_consolidator.run_consolidation_workflow.assert_called_once_with(
        session_id="session_1",
        decay_factor=0.95,
        pruning_threshold=0.1,
        similarity_threshold=0.85,
        forgetting_age_days=30.0
    )
    mock_save.assert_called_once_with(graph_store, vector_store, episodic_buffer)


@patch("cognigraph.cli.load_stores")
@patch("cognigraph.cli.save_stores")
def test_cmd_buffer_list(mock_save: MagicMock, mock_load: MagicMock, mock_stores: tuple[MagicMock, MagicMock, MagicMock]) -> None:
    """Tests the buffer list CLI command."""
    graph_store, vector_store, episodic_buffer = mock_stores
    episodic_buffer.get_messages.return_value = [
        StoredMessage(id="1", role="user", content="Hello", processed=False)
    ]

    args = argparse.Namespace(
        buffer_op="list",
        session_id="session_1",
        unprocessed_only=True,
        limit=10
    )
    cmd_buffer(args, graph_store, vector_store, episodic_buffer)

    episodic_buffer.get_messages.assert_called_once_with(
        session_id="session_1",
        unprocessed_only=True,
        limit=10
    )


@patch("cognigraph.cli.load_stores")
@patch("cognigraph.cli.save_stores")
def test_cmd_buffer_add(mock_save: MagicMock, mock_load: MagicMock, mock_stores: tuple[MagicMock, MagicMock, MagicMock]) -> None:
    """Tests the buffer add CLI command."""
    graph_store, vector_store, episodic_buffer = mock_stores

    args = argparse.Namespace(
        buffer_op="add",
        role="user",
        content="Hello world",
        session_id="session_1"
    )
    cmd_buffer(args, graph_store, vector_store, episodic_buffer)

    episodic_buffer.add_message.assert_called_once()
    mock_save.assert_called_once_with(graph_store, vector_store, episodic_buffer)


@patch("cognigraph.cli.load_stores")
@patch("cognigraph.cli.save_stores")
def test_cmd_buffer_clear(mock_save: MagicMock, mock_load: MagicMock, mock_stores: tuple[MagicMock, MagicMock, MagicMock]) -> None:
    """Tests the buffer clear CLI command."""
    graph_store, vector_store, episodic_buffer = mock_stores

    args = argparse.Namespace(buffer_op="clear")
    cmd_buffer(args, graph_store, vector_store, episodic_buffer)

    episodic_buffer.clear_processed.assert_called_once()
    mock_save.assert_called_once_with(graph_store, vector_store, episodic_buffer)


@patch("cognigraph.cli.load_stores")
def test_cmd_visualize(mock_load: MagicMock, mock_stores: tuple[MagicMock, MagicMock, MagicMock]) -> None:
    """Tests the visualize CLI command."""
    graph_store, vector_store, episodic_buffer = mock_stores
    graph_store.get_all_entities.return_value = [
        Entity(id="alice", name="Alice", type="Person")
    ]
    graph_store.get_all_relationships.return_value = [
        Relationship(source="alice", target="google", type="WORKS_AT")
    ]
    graph_store.get_degree.return_value = 1

    args = argparse.Namespace()
    cmd_visualize(args, graph_store, vector_store, episodic_buffer)

    mock_load.assert_called_once_with(graph_store, vector_store, episodic_buffer)
    graph_store.get_all_entities.assert_called_once()
    graph_store.get_all_relationships.assert_called_once()


@patch("cognigraph.cli.settings")
def test_init_components(mock_settings: MagicMock) -> None:
    """Tests component initialization in CLI."""
    mock_settings.USE_NEO4J = False
    mock_settings.VECTOR_STORE_TYPE = "simple"
    mock_settings.OPENAI_API_KEY = None
    mock_settings.EMBEDDING_DIMENSION = 64

    graph_store, vector_store, episodic_buffer, extractor, embedder = init_components()

    assert isinstance(graph_store, NetworkXGraphStore)
    assert isinstance(vector_store, SimpleVectorStore)
    assert isinstance(episodic_buffer, EpisodicBuffer)
    assert isinstance(extractor, RuleBasedExtractor)
    assert isinstance(embedder, MockEmbedder)
