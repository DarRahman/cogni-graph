# cognigraph/retriever.py - Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation and hierarchical entity-relation extraction.
# Contributed by Claude Code

"""Hybrid retriever combining vector similarity search and graph traversal."""

import logging
from typing import Dict, List, Set, Tuple
from cognigraph.graph_store import NetworkXGraphStore
from cognigraph.models import Entity, Relationship, RetrievalResult
from cognigraph.vector_store import SimpleVectorStore

logger = logging.getLogger("cognigraph.retriever")


class HybridRetriever:
    """Performs hybrid search: vector similarity to find entry points, then graph traversal for context."""

    def __init__(self, graph_store: NetworkXGraphStore, vector_store: SimpleVectorStore) -> None:
        """Initializes the retriever with graph and vector stores.

        Args:
            graph_store: The graph storage instance.
            vector_store: The vector storage instance.
        """
        self.graph_store = graph_store
        self.vector_store = vector_store
        logger.info("HybridRetriever initialized")

    def retrieve(self, query: str, query_vector: List[float], k: int = 3, depth: int = 1) -> RetrievalResult:
        """Retrieves a contextual subgraph relevant to the query.

        1. Finds top-k entities using vector similarity search.
        2. Traverses the graph up to `depth` steps from these seed entities.
        3. Collects all visited entities and relationships.

        Args:
            query: The query string.
            query_vector: The embedding vector of the query.
            k: Number of seed entities to retrieve via vector search.
            depth: Graph traversal depth from seed entities.

        Returns:
            A RetrievalResult containing the retrieved entities, relationships, and scores.
        """
        logger.info("Starting hybrid retrieval for query: '%s' (k=%d, depth=%d)", query, k, depth)

        # 1. Vector similarity search to find seed entities
        vector_results = self.vector_store.similarity_search(query_vector, k=k)
        seed_entity_ids = [res[0] for res in vector_results]
        scores = {res[0]: res[1] for res in vector_results}

        retrieved_entities: Dict[str, Entity] = {}
        retrieved_relationships: Set[Tuple[str, str, str]] = set()  # (source, target, type)
        relationships_list: List[Relationship] = []

        # Queue for BFS traversal: list of (entity_id, current_depth)
        queue = [(entity_id, 0) for entity_id in seed_entity_ids]
        visited = set(seed_entity_ids)

        # 2. Graph traversal
        while queue:
            curr_id, curr_depth = queue.pop(0)

            # Retrieve entity details
            entity = self.graph_store.get_entity(curr_id)
            if not entity:
                continue

            retrieved_entities[curr_id] = entity

            # If we haven't reached max depth, traverse neighbors
            if curr_depth < depth:
                neighbors = self.graph_store.get_neighbors(curr_id)
                for neighbor_entity, relationship in neighbors:
                    # Add neighbor to visited and queue if not visited
                    if neighbor_entity.id not in visited:
                        visited.add(neighbor_entity.id)
                        queue.append((neighbor_entity.id, curr_depth + 1))

                    # Add relationship to results if not already added
                    rel_key = (relationship.source, relationship.target, relationship.type)
                    if rel_key not in retrieved_relationships:
                        retrieved_relationships.add(rel_key)
                        relationships_list.append(relationship)

        logger.info(
            "Retrieval complete. Found %d entities and %d relationships",
            len(retrieved_entities),
            len(relationships_list)
        )

        return RetrievalResult(
            query=query,
            entities=list(retrieved_entities.values()),
            relationships=relationships_list,
            scores=scores
        )
