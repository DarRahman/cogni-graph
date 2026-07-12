# cognigraph/extractor.py - Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation and hierarchical entity-relation extraction.
# Contributed by Claude Code

"""Entity and relationship extraction engine."""

import logging
import re
from typing import Any, Dict, List, Optional, Protocol, Tuple
from pydantic import BaseModel, Field
from cognigraph.config import settings
from cognigraph.models import ChatMessage, Entity, ExtractionResult, Relationship

logger = logging.getLogger("cognigraph.extractor")


class Extractor(Protocol):
    """Protocol defining the interface for entity and relationship extractors."""

    def extract(self, messages: List[ChatMessage]) -> ExtractionResult:
        """Extracts entities and relationships from a list of chat messages.

        Args:
            messages: A list of ChatMessage objects to process.

        Returns:
            An ExtractionResult containing the extracted entities and relationships.
        """
        ...


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


class ExtractedEntity(BaseModel):
    """Pydantic model for entity extraction by the LLM."""

    name: str = Field(..., description="The name of the entity, capitalized and normalized (e.g., 'Google', 'John Doe')")
    type: str = Field(..., description="The category of the entity (e.g., Person, Organization, Location, Technology, Concept, Project)")
    description: str = Field(..., description="A short description of the entity and its role/context in the conversation")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Any additional key-value properties extracted for this entity (e.g., age, job title, location)")


class ExtractedRelationship(BaseModel):
    """Pydantic model for relationship extraction by the LLM."""

    source_entity_name: str = Field(..., description="The name of the source entity")
    target_entity_name: str = Field(..., description="The name of the target entity")
    type: str = Field(..., description="The type of relationship, in UPPERCASE_SNAKE_CASE (e.g., WORKS_AT, LIVES_IN, KNOWS, INTERESTED_IN, MEMBER_OF, OWNER_OF)")
    description: str = Field(..., description="A short description of the relationship context (e.g., 'John has been working at Google for 3 years')")
    weight: float = Field(1.0, description="The strength or confidence of the relationship (0.0 to 1.0)")


class ExtractedFacts(BaseModel):
    """Pydantic model for the complete set of extracted facts."""

    entities: List[ExtractedEntity] = Field(default_factory=list, description="List of entities extracted from the dialogue")
    relationships: List[ExtractedRelationship] = Field(default_factory=list, description="List of relationships extracted from the dialogue")


class InstructorExtractor:
    """An entity and relationship extractor using Instructor and OpenAI."""

    def __init__(self, client: Optional[Any] = None, model: Optional[str] = None) -> None:
        """Initializes the Instructor extractor.

        Args:
            client: Optional pre-configured instructor client. If None, creates one using settings.
            model: Optional LLM model name. If None, uses settings.LLM_MODEL.
        """
        self.model = model or settings.LLM_MODEL
        if client is not None:
            self.client = client
        else:
            import openai
            import instructor

            api_key = settings.OPENAI_API_KEY
            base_url = settings.OPENAI_API_BASE

            # Create standard OpenAI client
            openai_client = openai.OpenAI(api_key=api_key, base_url=base_url)
            # Patch it with instructor
            self.client = instructor.from_openai(openai_client)

        logger.info("InstructorExtractor initialized with model %s", self.model)

    def extract(self, messages: List[ChatMessage]) -> ExtractionResult:
        """Extracts entities and relationships from a list of chat messages using LLM.

        Args:
            messages: A list of ChatMessage objects to process.

        Returns:
            An ExtractionResult containing the extracted entities and relationships.
        """
        if not messages:
            logger.warning("No messages provided for extraction.")
            return ExtractionResult(entities=[], relationships=[])

        logger.info("Starting LLM extraction on %d messages", len(messages))

        # Format dialogue transcript
        formatted_transcript = "\n".join(
            f"{msg.role.upper()}: {msg.content}" for msg in messages
        )

        system_prompt = (
            "You are an advanced Entity-Relationship Extraction Engine. "
            "Your task is to analyze the conversation history between a User and an Assistant, "
            "and extract key entities (people, organizations, locations, concepts, projects, technologies) "
            "and the relationships between them.\n\n"
            "Guidelines:\n"
            "1. Extract all significant entities mentioned in the conversation. Provide a clear name, type, and description for each.\n"
            "2. Extract relationships between these entities. Relationships must be directed (source -> target) and have a specific type (e.g., WORKS_AT, LIVES_IN, KNOWS, INTERESTED_IN, MEMBER_OF, OWNER_OF, HAS_SKILL).\n"
            "3. For each relationship, provide a brief description explaining the context of the relationship as described in the dialogue.\n"
            "4. Assign a weight (0.0 to 1.0) to each relationship representing the strength or confidence of the relationship based on the dialogue.\n"
            "5. Extract any additional properties for entities (e.g., age, job title, location) and put them in the properties dictionary."
        )

        try:
            # Call LLM using instructor
            facts: ExtractedFacts = self.client.chat.completions.create(
                model=self.model,
                response_model=ExtractedFacts,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Extract facts from the following dialogue:\n\n{formatted_transcript}"}
                ]
            )

            # Map ExtractedFacts to ExtractionResult
            result = self._map_facts(facts)
            logger.info(
                "LLM extraction complete. Extracted %d entities and %d relationships",
                len(result.entities),
                len(result.relationships)
            )
            return result

        except Exception as e:
            logger.exception("Failed to extract facts using LLM")
            raise e

    def _map_facts(self, facts: ExtractedFacts) -> ExtractionResult:
        """Maps ExtractedFacts to the internal ExtractionResult model."""
        entities_dict: Dict[str, Entity] = {}
        relationships: List[Relationship] = []

        # 1. Map entities
        for ext_entity in facts.entities:
            entity_id = ext_entity.name.lower().strip()
            if not entity_id:
                continue

            entities_dict[entity_id] = Entity(
                id=entity_id,
                name=ext_entity.name.strip(),
                type=ext_entity.type.strip(),
                description=ext_entity.description.strip(),
                properties=ext_entity.properties or {}
            )

        # 2. Map relationships
        for ext_rel in facts.relationships:
            source_id = ext_rel.source_entity_name.lower().strip()
            target_id = ext_rel.target_entity_name.lower().strip()

            if not source_id or not target_id:
                continue

            # Ensure source and target entities exist in our entities dict
            if source_id not in entities_dict:
                entities_dict[source_id] = Entity(
                    id=source_id,
                    name=ext_rel.source_entity_name.strip(),
                    type="Concept",
                    description=f"Auto-created source entity for relationship: '{ext_rel.source_entity_name}'"
                )
            if target_id not in entities_dict:
                entities_dict[target_id] = Entity(
                    id=target_id,
                    name=ext_rel.target_entity_name.strip(),
                    type="Concept",
                    description=f"Auto-created target entity for relationship: '{ext_rel.target_entity_name}'"
                )

            relationships.append(
                Relationship(
                    source=source_id,
                    target=target_id,
                    type=ext_rel.type.strip().upper(),
                    description=ext_rel.description.strip(),
                    weight=max(0.0, min(1.0, ext_rel.weight))
                )
            )

        return ExtractionResult(
            entities=list(entities_dict.values()),
            relationships=relationships
        )
