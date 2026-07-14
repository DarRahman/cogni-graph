# cognigraph/pipeline.py - Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation and hierarchical entity-relation extraction.
# Contributed by Claude Code

"""Consolidation pipeline orchestrating ingestion, extraction, and memory consolidation."""

import hashlib
import logging
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import networkx as nx
from cognigraph.config import settings
from cognigraph.extractor import Extractor
from cognigraph.graph_store import GraphStore
from cognigraph.models import ChatMessage, Entity, ExtractionResult, Relationship
from cognigraph.vector_store import VectorStore
from cognigraph.episodic_buffer import EpisodicBuffer

logger = logging.getLogger("cognigraph.pipeline")


def jaro_winkler_similarity(s1: str, s2: str) -> float:
    """Calculates the Jaro-Winkler similarity between two strings.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        A float between 0.0 and 1.0 representing similarity.
    """
    s1 = s1.strip().lower()
    s2 = s2.strip().lower()
    
    len1 = len(s1)
    len2 = len(s2)
    
    if len1 == 0 or len2 == 0:
        return 0.0
        
    if s1 == s2:
        return 1.0
        
    # Maximum distance for matching characters
    match_bound = max(len1, len2) // 2 - 1
    if match_bound < 0:
        match_bound = 0
        
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    
    matches = 0
    transpositions = 0
    
    for i in range(len1):
        start = max(0, i - match_bound)
        end = min(len2, i + match_bound + 1)
        for j in range(start, end):
            if not s2_matches[j] and s1[i] == s2[j]:
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break
                
    if matches == 0:
        return 0.0
        
    # Count transpositions
    k = 0
    for i in range(len1):
        if s1_matches[i]:
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1
            
    transpositions //= 2
    
    # Jaro similarity
    jaro = (matches / len1 + matches / len2 + (matches - transpositions) / matches) / 3.0
    
    # Winkler modification
    # Find length of common prefix up to 4 characters
    prefix_len = 0
    for i in range(min(4, len1, len2)):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break
            
    return jaro + prefix_len * 0.1 * (1.0 - jaro)


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculates the cosine similarity between two vectors.

    Args:
        v1: First vector.
        v2: Second vector.

    Returns:
        Cosine similarity score.
    """
    arr1 = np.array(v1)
    arr2 = np.array(v2)
    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(arr1, arr2) / (norm1 * norm2))


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
        graph_store: GraphStore,
        vector_store: VectorStore,
        extractor: Extractor,
        embedder: MockEmbedder,
        episodic_buffer: Optional[EpisodicBuffer] = None
    ) -> None:
        """Initializes the pipeline.

        Args:
            graph_store: The graph storage instance.
            vector_store: The vector storage instance conforming to VectorStore protocol.
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
        3. Prunes weak relationships and isolated entities.
        4. Resolves duplicates and merges highly similar entities.
        """
        logger.info("Starting memory consolidation loop")

        # 1. Process any unprocessed messages in the episodic buffer
        if self.episodic_buffer:
            unprocessed = self.episodic_buffer.get_messages(unprocessed_only=True)
            if unprocessed:
                logger.info("Consolidating %d unprocessed messages from episodic buffer", len(unprocessed))
                chat_messages = [
                    ChatMessage(
                        role=msg.role,
                        content=msg.content,
                        timestamp=msg.timestamp,
                        metadata=msg.metadata
                    )
                    for msg in unprocessed
                ]
                # Extract and process directly to avoid re-adding to episodic buffer
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
                
                # Mark original messages as processed
                self.episodic_buffer.mark_as_processed([msg.id for msg in unprocessed])

        # 2. Decay and prune relationships
        decay_factor = settings.RECENCY_DECAY_FACTOR
        pruning_threshold = settings.PRUNING_THRESHOLD
        
        # Check if the store has an optimized implementation
        if hasattr(self.graph_store, "decay_and_prune"):
            logger.info("Running optimized decay and pruning on graph store")
            self.graph_store.decay_and_prune(decay_factor, pruning_threshold)  # type: ignore[attr-defined]
        else: 
            logger.info("Running generic decay and pruning on graph store")
            relationships = self.graph_store.get_all_relationships()
            now = datetime.utcnow()

            for rel in relationships:
                elapsed_time = now - rel.updated_at
                elapsed_days = elapsed_time.total_seconds() / 86400.0
                
                # Decay weight
                new_weight = rel.weight * (decay_factor ** elapsed_days)
                
                if new_weight < pruning_threshold:
                    self.graph_store.remove_relationship(rel.source, rel.target, rel.type)
                    logger.info("Pruned relationship: %s -[%s]-> %s (weight %f decayed to %f)", 
                                rel.source, rel.type, rel.target, rel.weight, new_weight)
                else:
                    # Remove first to avoid incrementing the weight in add_relationship
                    self.graph_store.remove_relationship(rel.source, rel.target, rel.type)
                    rel.weight = new_weight
                    rel.updated_at = now
                    self.graph_store.add_relationship(rel)

            # Prune isolated nodes
            entities = self.graph_store.get_all_entities()
            for entity in entities:
                if self.graph_store.get_degree(entity.id) == 0:
                    self.graph_store.remove_entity(entity.id)
                    self.vector_store.delete_vector(entity.id)
                    logger.info("Pruned isolated entity: %s", entity.id)

        # 3. Entity Resolution & Merging
        entities = self.graph_store.get_all_entities()
        merge_pairs = []
        
        for i, ent1 in enumerate(entities):
            for j, ent2 in enumerate(entities):
                if i >= j:
                    continue
                
                if not self._are_types_compatible(ent1.type, ent2.type):
                    continue
                
                str_sim = jaro_winkler_similarity(ent1.name, ent2.name)
                
                vec_sim = 0.0
                v1 = self.vector_store.get_vector(ent1.id)
                v2 = self.vector_store.get_vector(ent2.id)
                if v1 is not None and v2 is not None:
                    vec_sim = cosine_similarity(v1, v2)
                
                # Merge criteria:
                # - Extremely high string similarity (>= 0.95)
                # - High vector similarity (>= SIMILARITY_THRESHOLD) AND reasonable string similarity (>= 0.75)
                should_merge = False
                if str_sim >= 0.95:
                    should_merge = True
                elif vec_sim >= settings.SIMILARITY_THRESHOLD and str_sim >= 0.75:
                    should_merge = True
                    
                if should_merge:
                    merge_pairs.append((ent1.id, ent2.id))
                    logger.info("Identified duplicate pair: %s (%s) and %s (%s) [str_sim=%.2f, vec_sim=%.2f]", 
                                ent1.id, ent1.name, ent2.id, ent2.name, str_sim, vec_sim)

        if merge_pairs:
            # Build a temporary graph to find connected components (clusters of duplicates)
            merge_graph = nx.Graph()
            merge_graph.add_nodes_from([e.id for e in entities])
            for id1, id2 in merge_pairs:
                merge_graph.add_edge(id1, id2)
            
            clusters = [list(c) for c in nx.connected_components(merge_graph) if len(c) > 1]
            
            for cluster in clusters:
                # Sort cluster to find the primary entity
                # Criteria: 1. Degree (highest first), 2. Description length (longest first), 3. Age (oldest first)
                def sort_key(ent_id: str) -> Tuple[int, int, float]:
                    ent = self.graph_store.get_entity(ent_id)
                    if not ent:
                        return (0, 0, 0.0)
                    deg = self.graph_store.get_degree(ent_id)
                    desc_len = len(ent.description) if ent.description else 0
                    created_ts = ent.created_at.timestamp()
                    return (-deg, -desc_len, created_ts)
                
                cluster.sort(key=sort_key)
                primary_id = cluster[0]
                duplicate_ids = cluster[1:]
                
                logger.info("Merging cluster %s into primary entity %s", duplicate_ids, primary_id)
                self._merge_cluster(primary_id, duplicate_ids)

        logger.info("Consolidation complete")

    def _are_types_compatible(self, type1: str, type2: str) -> bool:
        """Checks if two entity types are compatible for merging.

        Args:
            type1: First entity type.
            type2: Second entity type.

        Returns:
            True if compatible, else False.
        """
        t1 = type1.strip().lower()
        t2 = type2.strip().lower()
        
        if t1 == t2:
            return True
            
        # 'concept' is a generic type and can be merged with any specific type
        if t1 == "concept" or t2 == "concept":
            return True
            
        return False

    def _merge_cluster(self, primary_id: str, duplicate_ids: List[str]) -> None:
        """Merges a list of duplicate entities into a primary entity.

        Args:
            primary_id: The ID of the primary entity.
            duplicate_ids: The IDs of the duplicate entities to merge.
        """
        primary = self.graph_store.get_entity(primary_id)
        if not primary:
            return
            
        merged_properties = primary.properties.copy()
        merged_descriptions = [primary.description] if primary.description else []
        
        created_at = primary.created_at
        updated_at = primary.updated_at
        
        for dup_id in duplicate_ids:
            dup = self.graph_store.get_entity(dup_id)
            if not dup:
                continue
                
            # Merge properties: newer values overwrite older ones
            if dup.updated_at > updated_at:
                merged_properties.update(dup.properties)
            else:
                for k, v in dup.properties.items():
                    if k not in merged_properties:
                        merged_properties[k] = v
                        
            if dup.description and dup.description not in merged_descriptions:
                merged_descriptions.append(dup.description)
                
            created_at = min(created_at, dup.created_at)
            updated_at = max(updated_at, dup.updated_at)
            
        # Combine descriptions
        combined_description = " | ".join(merged_descriptions)
        
        # Resolve type: if primary is a concept, try to find a more specific type from duplicates
        resolved_type = primary.type
        if resolved_type.lower() == "concept":
            for dup_id in duplicate_ids:
                dup_ent = self.graph_store.get_entity(dup_id)
                if dup_ent and dup_ent.type.lower() != "concept":
                    resolved_type = dup_ent.type
                    break
        
        # Create the merged entity
        merged_entity = Entity(
            id=primary_id,
            name=primary.name,
            type=resolved_type,
            description=combined_description,
            properties=merged_properties,
            created_at=created_at,
            updated_at=updated_at
        )
        
        # Merge in graph store
        for dup_id in duplicate_ids:
            self.graph_store.merge_entities(primary_id, dup_id, merged_entity)
            
            # Remove duplicate from vector store
            self.vector_store.delete_vector(dup_id)
                    
        # Update primary vector with new description
        embedding_text = f"{merged_entity.name}: {merged_entity.description}"
        vector = self.embedder.embed_text(embedding_text)
        self.vector_store.add_vector(
            vector_id=primary_id,
            vector=vector,
            metadata={"name": merged_entity.name, "type": merged_entity.type}
        )
