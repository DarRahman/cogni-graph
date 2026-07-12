# tests/test_extractor.py - Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation and hierarchical entity-relation extraction.
# Contributed by Claude Code

"""Unit tests for the entity and relationship extractor."""

from datetime import datetime
from unittest.mock import MagicMock
from cognigraph.extractor import (
    ExtractedEntity,
    ExtractedFacts,
    ExtractedRelationship,
    InstructorExtractor,
    RuleBasedExtractor,
)
from cognigraph.models import ChatMessage


def test_rule_based_extractor_relations() -> None:
    """Tests that the extractor correctly identifies relationships and entities from text."""
    extractor = RuleBasedExtractor()

    messages = [
        ChatMessage(
            role="user",
            content="Alice works at Google. She likes Python.",
            timestamp=datetime.utcnow()
        )
    ]

    result = extractor.extract(messages)

    # Verify entities extracted
    entity_ids = {entity.id for entity in result.entities}
    assert "alice" in entity_ids
    assert "google" in entity_ids
    assert "python" in entity_ids

    # Verify relationship extracted
    assert len(result.relationships) >= 2
    rel_types = {rel.type for rel in result.relationships}
    assert "WORKS_AT" in rel_types
    assert "LIKES" in rel_types

    # Verify specific relationship details
    works_at_rel = next(rel for rel in result.relationships if rel.type == "WORKS_AT")
    assert works_at_rel.source == "alice"
    assert works_at_rel.target == "google"


def test_rule_based_extractor_standalone_entities() -> None:
    """Tests that standalone capitalized words are extracted as fallback entities."""
    extractor = RuleBasedExtractor()

    messages = [
        ChatMessage(
            role="user",
            content="Yesterday I visited Paris and Berlin.",
            timestamp=datetime.utcnow()
        )
    ]

    result = extractor.extract(messages)

    entity_ids = {entity.id for entity in result.entities}
    assert "paris" in entity_ids
    assert "berlin" in entity_ids


def test_instructor_extractor_mock() -> None:
    """Tests InstructorExtractor using a mocked client to avoid real API calls."""
    mock_client = MagicMock()

    # Define the expected return value from the mocked instructor client
    expected_facts = ExtractedFacts(
        entities=[
            ExtractedEntity(
                name="Alice",
                type="Person",
                description="A software engineer",
                properties={"role": "developer"}
            ),
            ExtractedEntity(
                name="Google",
                type="Organization",
                description="A tech company",
                properties={}
            )
        ],
        relationships=[
            ExtractedRelationship(
                source_entity_name="Alice",
                target_entity_name="Google",
                type="WORKS_AT",
                description="Alice works at Google",
                weight=0.9
            )
        ]
    )

    # Mock the chat.completions.create call
    mock_client.chat.completions.create.return_value = expected_facts

    # Initialize extractor with mock client
    extractor = InstructorExtractor(client=mock_client, model="mock-model")

    messages = [
        ChatMessage(role="user", content="Alice works at Google.")
    ]

    result = extractor.extract(messages)

    # Verify mock was called correctly
    mock_client.chat.completions.create.assert_called_once()
    _, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["model"] == "mock-model"
    assert kwargs["response_model"] == ExtractedFacts

    # Verify mapping
    assert len(result.entities) == 2
    entity_ids = {e.id for e in result.entities}
    assert "alice" in entity_ids
    assert "google" in entity_ids

    alice = next(e for e in result.entities if e.id == "alice")
    assert alice.name == "Alice"
    assert alice.type == "Person"
    assert alice.properties == {"role": "developer"}

    assert len(result.relationships) == 1
    rel = result.relationships[0]
    assert rel.source == "alice"
    assert rel.target == "google"
    assert rel.type == "WORKS_AT"
    assert rel.weight == 0.9
