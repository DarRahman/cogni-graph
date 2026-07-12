# cognigraph/extractor.py - Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation and hierarchical entity-relation extraction.
# Contributed by Claude Code

"""Entity and relationship extraction engine."""

import logging
import re
from typing import Dict, List, Tuple
from cognigraph.models import ChatMessage, Entity, ExtractionResult, Relationship

logger = logging.getLogger("cognigraph.extractor")


class RuleBasedExtractor:
    """A rule-based entity and relationship extractor for bootstrapping and testing.

    Extracts entities based on capitalized words and relationships based on common verb patterns.
    """

    def __init__(self) -> None:
        """Initializes the extractor with basic patterns."""
        # Simple patterns for relationship extraction: "Entity1 verb Entity2"
        self.relation_patterns: List[Tuple[re.Pattern[str], str]] = [
            (re.compile(r"\b([A-Z][a-z]+)\s+(works\s+at|is\s+employed\s+by|joined)\s+([A-Z][a-z]+)\b", re.IGNORECASE), "WORKS_AT"),
            (re.compile(r"\b([A-Z][a-z]+)\s+(likes|loves|enjoys|prefers)\s+([A-Z][a-z]+)\b", re.IGNORECASE), "LIKES"),
            (re.compile(r"\b([A-Z][a-z]+)\s+(is\s+a|is\s+an|is)\s+([A-Z][a-z]+)\b", re.IGNORECASE), "IS_A"),
            (re.compile(r"\b([A-Z][a-z]+)\s+(lives\s+in|resides\s+in|visited)\s+([A-Z][a-z]+)\b", re.IGNORECASE), "LIVES_IN"),
            (re.compile(r"\b([A-Z][a-z]+)\s+(knows|met|collaborates\s+with)\s+([A-Z][a-z]+)\b", re.IGNORECASE), "KNOWS"),
        ]
        logger.info("RuleBasedExtractor initialized with %d patterns", len(self.relation_patterns))

    def extract(self, messages: List[ChatMessage]) -> ExtractionResult:
        """Extracts entities and relationships from a list of chat messages.

        Args:
            messages: A list of ChatMessage objects to process.

        Returns:
            An ExtractionResult containing the extracted entities and relationships.
        """
        logger.info("Starting extraction on %d messages", len(messages))
        entities_dict: Dict[str, Entity] = {}
        relationships = []

        for msg in messages:
            text = msg.content
            # 1. Extract relationships using patterns
            for pattern, rel_type in self.relation_patterns:
                for match in pattern.finditer(text):
                    source_name = match.group(1).strip()
                    target_name = match.group(3).strip()
                    rel_desc = match.group(0).strip()

                    # Normalize IDs
                    source_id = source_name.lower()
                    target_id = target_name.lower()

                    # Create entities if they don't exist
                    if source_id not in entities_dict:
                        entities_dict[source_id] = Entity(
                            id=source_id,
                            name=source_name,
                            type="Concept" if rel_type == "IS_A" else "Person",
                            description=f"Entity extracted from text: '{source_name}'"
                        )
                    if target_id not in entities_dict:
                        entities_dict[target_id] = Entity(
                            id=target_id,
                            name=target_name,
                            type="Organization" if rel_type == "WORKS_AT" else "Concept",
                            description=f"Entity extracted from text: '{target_name}'"
                        )

                    # Create relationship
                    relationship = Relationship(
                        source=source_id,
                        target=target_id,
                        type=rel_type,
                        description=rel_desc,
                        weight=1.0
                    )
                    relationships.append(relationship)
                    logger.debug("Extracted relationship: %s -> %s (%s)", source_id, target_id, rel_type)

            # 2. Fallback: Extract standalone capitalized words as entities if no relationships found
            # (only words with length >= 3 to avoid short acronyms/noise)
            words = re.findall(r"\b[A-Z][a-z]+\b", text)
            for word in words:
                word_id = word.lower()
                if len(word) >= 3 and word_id not in entities_dict:
                    entities_dict[word_id] = Entity(
                        id=word_id,
                        name=word,
                        type="Concept",
                        description=f"Standalone entity extracted: '{word}'"
                    )
                    logger.debug("Extracted standalone entity: %s", word_id)

        logger.info("Extraction complete. Extracted %d entities and %d relationships", len(entities_dict), len(relationships))
        return ExtractionResult(
            entities=list(entities_dict.values()),
            relationships=relationships
        )
