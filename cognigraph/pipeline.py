# cognigraph/pipeline.py - Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation and hierarchical entity-relation extraction.
# Contributed by Claude Code

"""Consolidation pipeline orchestrating ingestion, extraction, and memory consolidation."""

import hashlib
import logging
from datetime import datetime
from typing import List, Optional
import numpy as np
from cognigraph.config import settings
from cognigraph.extractor import Extractor
from cognigraph.graph_store import NetworkXGraphStore
from cognigraph.models import ChatMessage, Entity, ExtractionResult
from cognigraph.vector_store import SimpleVectorStore
from cognigraph.episodic_buffer import EpisodicBuffer

logger = logging.getLogger("cognigraph.pipeline")


class MockEmbedder:
    """Generates deterministic mock embeddings for testing and bootstrapping."""

    def __init__(self, dimension: int = 1536) -> None:
        """Initializes the mock embedder.

        Args:
            dimension: The dimension of the embedding vector.
        """
        self.dimension = dimension

    def embed_text(self, text: str) -> List[float]:
        """Generates a deterministic unit vector based on the MD5 hash of the text.

        Args:
            text: The input text to embed.

        Returns:
            A list of floats representing the unit vector.
        """
        # Seed numpy random generator with hash of text for determinism
        hasher = hashlib.md5(text.encode("utf-8"))
        seed = int(hasher.hexdigest(), 16) % (2**32 - 1)
        rng = np.random.default_rng(seed)

        # Generate random vector and normalize to unit length
        vec = rng.normal(size=self.dimension)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


class ConsolidationPipeline:
    """Orchestrates the ingestion, extraction, storage, and consolidation of memory."""

    def __init__(
        self,
        graph_store: NetworkXGraphStore,
        vector_store: SimpleVectorStore,
        extractor: Extractor,
        embedder: MockEmbedder,
        episodic_buffer: Optional[EpisodicBuffer] = None
    ) -> None:
        """Initializes the pipeline.

        Args:
            graph_store: The graph storage instance.
            vector_store: The vector storage instance.
            extractor: The entity-relationship extractor.
            embedder: The embedding generator.
            episodic_buffer: Optional episodic buffer instance.
        """
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.extractor = extractor
        self.embedder = embedder
        self.episodic_buffer = episodic_buffer
        logger.info("ConsolidationPipeline initialized")

    def ingest_and_process(self, messages: List[ChatMessage]) -> ExtractionResult:
        """Ingests chat messages, extracts facts, and updates stores.

        Args:
            messages: A list of ChatMessage objects.

        Returns:
            The ExtractionResult containing the extracted entities and relationships.
        """
        logger.info("Processing %d messages in pipeline", len(messages))

        # If episodic buffer is available, store them first
        stored_ids = []
        if self.episodic_buffer:
            stored_ids = self.episodic_buffer.add_messages(messages)

        # 1. Extract entities and relationships
        extraction_result = self.extractor.extract(messages)

        # 2. Add entities to graph and vector stores
        for entity in extraction_result.entities:
            self.graph_store.add_entity(entity)

            # Generate embedding for the entity name/description
            embedding_text = f"{entity.name}: {entity.description}"
            vector = self.embedder.embed_text(embedding_text)
            self.vector_store.add_vector(
                vector_id=entity.id,
                vector=vector,
                metadata={"name": entity.name, "type": entity.type}
            )

        # 3. Add relationships to graph store
        for relationship in extraction_result.relationships:
            self.graph_store.add_relationship(relationship)

        # Mark as processed in episodic buffer
        if self.episodic_buffer and stored_ids:
            self.episodic_buffer.mark_as_processed(stored_ids)

        logger.info("Ingestion and processing complete")
        return extraction_result

    def consolidate(self) -> None:
        """Performs memory consolidation.

        1. Consolidates unprocessed messages from the episodic buffer if available.
        2. Decays relationship weights based on recency.
        3. Merges highly similar entities (based on name similarity or vector similarity).
        """
        logger.info("Starting memory consolidation loop")

        # 1. Process any unprocessed messages in the episodic buffer
        if self.episodic_buffer:
            unprocessed = self.episodic_buffer.get_messages(unprocessed_only=True)
            if unprocessed:
                logger.info("Consolidating %d unprocessed messages from episodic buffer", len(unprocessed))
                # Convert StoredMessage to ChatMessage for processing
                chat_messages = [
                    ChatMessage(
                        role=msg.role,
                        content=msg.content,
                        timestamp=msg.timestamp,
                        metadata=msg.metadata
                    )
                    for msg in unprocessed
                ]
                # Extract and store facts directly without re-adding to episodic buffer
                extraction_result = self.extractor.extract(chat_messages)

                for entity in extraction_result.entities:
                    self.graph_store.add_entity(entity)
                    embedding_text = f"{entity.name}: {entity.description}"
                    vector = self.embedder.embed_text(embedding_text)
                    self.vector_store.add_vector(
                        vector_id=entity.id,
                        vector=vector,
                        metadata={"name": entity.name, "type": entity.type}
                    )

                for relationship in extraction_result.relationships:
                    self.graph_store.add_relationship(relationship)

                # Mark the original messages as processed
                msg_ids = [msg.id for msg in unprocessed]
                self.episodic_buffer.mark_as_processed(msg_ids)

        # 2. Decay relationship weights
        decay_factor = settings.RECENCY_DECAY_FACTOR
        for u, v, key, data in list(self.graph_store.graph.edges(keys=True, data=True)):
            current_weight = data.get("weight", 1.0)
            new_weight = current_weight * decay_factor
            self.graph_store.graph[u][v][key]["weight"] = new_weight
            self.graph_store.graph[u][v][key]["updated_at"] = datetime.utcnow().isoformat()
            logger.debug("Decayed relationship weight: %s -[%s]-> %s to %f", u, key, v, new_weight)

        # 3. Merge duplicate entities (simple name-based consolidation for scaffolding)
        nodes = list(self.graph_store.graph.nodes(data=True))
        merged_count = 0

        for i, (node_id_1, attrs_1) in enumerate(nodes):
            for j, (node_id_2, attrs_2) in enumerate(nodes):
                if i >= j:
                    continue

                # Check if they are already deleted/merged in this loop
                if not self.graph_store.graph.has_node(node_id_1) or not self.graph_store.graph.has_node(node_id_2):
                    continue

                name_1 = attrs_1.get("name", "").lower()
                name_2 = attrs_2.get("name", "").lower()

                # Simple exact match or containment check for scaffolding
                if name_1 == name_2 or (len(name_1) > 3 and len(name_2) > 3 and (name_1 in name_2 or name_2 in name_1)):
                    # Merge node_2 into node_1
                    merged_entity = Entity(
                        id=node_id_1,
                        name=attrs_1.get("name", node_id_1),
                        type=attrs_1.get("type", "Concept"),
                        description=f"{attrs_1.get('description', '')} | Merged with: {attrs_2.get('description', '')}",
                        properties={**attrs_1.get("properties", {}), **attrs_2.get("properties", {})}
                    )
                    self.graph_store.merge_entities(node_id_1, node_id_2, merged_entity)
                    merged_count += 1

        logger.info("Consolidation complete. Merged %d entities", merged_count)
