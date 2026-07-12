# cognigraph/models.py - Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation and hierarchical entity-relation extraction.
# Contributed by Claude Code

"""Data models for CogniGraph memory engine."""

from datetime import datetime
from typing import Any, Dict, List
from pydantic import BaseModel, Field


class Entity(BaseModel):
    """Represents a node in the semantic knowledge graph."""

    id: str = Field(..., description="Unique identifier for the entity, typically normalized name")
    name: str = Field(..., description="Display name of the entity")
    type: str = Field(..., description="Category of the entity (e.g., Person, Organization, Concept, Project)")
    description: str = Field("", description="Summary description of the entity")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary key-value properties")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when entity was first created")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of last update")


class Relationship(BaseModel):
    """Represents a directed edge between two entities in the knowledge graph."""

    source: str = Field(..., description="ID of the source entity")
    target: str = Field(..., description="ID of the target entity")
    type: str = Field(..., description="Type of relationship (e.g., WORKS_FOR, INTERESTED_IN, CREATED)")
    description: str = Field("", description="Contextual description of the relationship")
    weight: float = Field(1.0, description="Strength or frequency of the relationship")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary key-value properties")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when relationship was first created")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of last update")


class ChatMessage(BaseModel):
    """Represents a single message in a conversation stream."""

    role: str = Field(..., description="Role of the speaker (e.g., user, assistant, system)")
    content: str = Field(..., description="Text content of the message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of the message")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata (e.g., session_id, user_id)")


class ExtractionResult(BaseModel):
    """Represents the structured facts extracted from a conversation segment."""

    entities: List[Entity] = Field(default_factory=list, description="Extracted entities")
    relationships: List[Relationship] = Field(default_factory=list, description="Extracted relationships between entities")


class RetrievalResult(BaseModel):
    """Represents the result of a hybrid vector-graph query."""

    query: str = Field(..., description="The original query string")
    entities: List[Entity] = Field(default_factory=list, description="Relevant entities found")
    relationships: List[Relationship] = Field(default_factory=list, description="Relevant relationships connecting the entities")
    scores: Dict[str, float] = Field(default_factory=dict, description="Relevance scores for retrieved entities")
