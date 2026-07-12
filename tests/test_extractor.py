# tests/test_extractor.py - Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation and hierarchical entity-relation extraction.
# Contributed by Claude Code

"""Unit tests for the entity and relationship extractor."""

from datetime import datetime
from cognigraph.extractor import RuleBasedExtractor
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
