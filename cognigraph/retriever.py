# cognigraph/retriever.py - Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation and hierarchical entity-relation extraction.
# Contributed by Claude Code

"""Hybrid retriever combining vector similarity search and graph traversal."""

import logging
from typing import Dict, List, Set, Tuple, Optional
from cognigraph.graph_store import GraphStore
from cognigraph.vector_store import VectorStore
from cognigraph.models import Entity, Relationship, RetrievalResult

logger = logging.getLogger("cognigraph.retriever")


class HybridRetriever:
    """Performs hybrid search: vector similarity to find entry points, then graph traversal for context.

    Uses advanced algorithms like Local Personalized PageRank (Random Walk with Restart)
    and Reciprocal Rank Fusion (RRF) to combine vector and graph relevance.
    """

    def __init__(
        self,
        graph_store: GraphStore,
        vector_store: VectorStore,
        alpha: float = 0.15,
        beta: float = 0.5,
        use_rrf: bool = True,
        rrf_k: int = 60
    ) -> None:
        """Initializes the hybrid retriever.

        Args:
            graph_store: The graph storage instance conforming to GraphStore protocol.
            vector_store: The vector storage instance conforming to VectorStore protocol.
            alpha: Restart probability for Personalized PageRank (0.0 to 1.0).
            beta: Weight for vector similarity in linear combination (0.0 to 1.0).
                  Only used if use_rrf is False.
            use_rrf: If True, uses Reciprocal Rank Fusion (RRF) to combine rankings.
                     If False, uses a weighted linear combination of normalized scores.
            rrf_k: Constant parameter for RRF (typically 60).
        """
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.alpha = alpha
        self.beta = beta
        self.use_rrf = use_rrf
        self.rrf_k = rrf_k
        logger.info(
            "HybridRetriever initialized (alpha=%.2f, beta=%.2f, use_rrf=%s, rrf_k=%d)",
            alpha, beta, use_rrf, rrf_k
        )

    def retrieve(
        self,
        query: str,
        query_vector: List[float],
        k: int = 3,
        depth: int = 2,
        max_nodes: int = 10,
        include_types: Optional[List[str]] = None,
        exclude_types: Optional[List[str]] = None
    ) -> RetrievalResult:
        """Retrieves a contextual subgraph relevant to the query.

        1. Finds top-k seed entities using vector similarity search.
        2. Performs a local Personalized PageRank (Random Walk with Restart) from the seeds.
        3. Combines vector similarity and graph relevance scores.
        4. Selects the top `max_nodes` entities and extracts the connecting relationships.

        Args:
            query: The query string.
            query_vector: The embedding vector of the query.
            k: Number of seed entities to retrieve via vector search.
            depth: Graph traversal depth from seed entities.
            max_nodes: Maximum number of entities to return in the final result.
            include_types: Optional list of entity types to include.
            exclude_types: Optional list of entity types to exclude.

        Returns:
            A RetrievalResult containing the retrieved entities, relationships, and scores.
        """
        logger.info(
            "Starting hybrid retrieval for query: '%s' (k=%d, depth=%d, max_nodes=%d)",
            query, k, depth, max_nodes
        )

        # 1. Vector similarity search to find seed entities
        vector_results = self.vector_store.similarity_search(query_vector, k=k)
        if not vector_results:
            logger.warning("Vector search returned no results for query: '%s'", query)
            return RetrievalResult(query=query, entities=[], relationships=[], scores={})

        seed_ids = [res[0] for res in vector_results]
        vector_scores = {res[0]: res[1] for res in vector_results}

        # Normalize vector scores to sum to 1.0 for personalization vector
        total_vector_score = sum(vector_scores.values())
        if total_vector_score > 0:
            personalization = {node_id: score / total_vector_score for node_id, score in vector_scores.items()}
        else:
            personalization = {node_id: 1.0 / len(seed_ids) for node_id in seed_ids}

        # 2. Get local subgraph and run PPR
        local_nodes, adjacency = self._get_local_subgraph(seed_ids, depth)
        ppr_scores = self._compute_local_ppr(local_nodes, adjacency, personalization)

        # 3. Combine scores using RRF or Weighted Linear Combination
        combined_scores: Dict[str, float] = {}

        if self.use_rrf:
            # Rank by vector similarity
            sorted_vector = sorted(vector_scores.items(), key=lambda x: x[1], reverse=True)
            vector_ranks = {node_id: rank + 1 for rank, (node_id, _) in enumerate(sorted_vector)}

            # Rank by PPR score
            sorted_ppr = sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True)
            ppr_ranks = {node_id: rank + 1 for rank, (node_id, _) in enumerate(sorted_ppr)}

            for node_id in local_nodes:
                v_rank = vector_ranks.get(node_id)
                p_rank = ppr_ranks.get(node_id)

                v_score = 1.0 / (self.rrf_k + v_rank) if v_rank is not None else 0.0
                p_score = 1.0 / (self.rrf_k + p_rank) if p_rank is not None else 0.0

                combined_scores[node_id] = v_score + p_score
        else:
            # Normalize vector scores to [0, 1]
            max_v = max(vector_scores.values()) if vector_scores else 1.0
            min_v = min(vector_scores.values()) if vector_scores else 0.0
            range_v = max_v - min_v if max_v != min_v else 1.0

            # Normalize PPR scores to [0, 1]
            max_p = max(ppr_scores.values()) if ppr_scores else 1.0
            min_p = min(ppr_scores.values()) if ppr_scores else 0.0
            range_p = max_p - min_p if max_p != min_p else 1.0

            for node_id in local_nodes:
                v_score = vector_scores.get(node_id, 0.0)
                norm_v = (v_score - min_v) / range_v if node_id in vector_scores else 0.0

                p_score = ppr_scores.get(node_id, 0.0)
                norm_p = (p_score - min_p) / range_p

                combined_scores[node_id] = self.beta * norm_v + (1.0 - self.beta) * norm_p

        # 4. Filter and sort entities
        filtered_entities: List[Entity] = []
        for node_id in local_nodes:
            entity = self.graph_store.get_entity(node_id)
            if not entity:
                continue

            # Filter by type
            if include_types and entity.type not in include_types:
                continue
            if exclude_types and entity.type in exclude_types:
                continue

            filtered_entities.append(entity)

        # Sort by combined score descending
        filtered_entities.sort(key=lambda e: combined_scores.get(e.id, 0.0), reverse=True)

        # Limit to max_nodes
        top_entities = filtered_entities[:max_nodes]
        top_entity_ids = {e.id for e in top_entities}

        # 5. Extract relationships connecting the top entities
        retrieved_relationships: Set[Tuple[str, str, str]] = set()
        relationships_list: List[Relationship] = []

        for entity_id in top_entity_ids:
            neighbors = self.graph_store.get_neighbors(entity_id)
            for neighbor_entity, relationship in neighbors:
                if neighbor_entity.id in top_entity_ids:
                    rel_key = (relationship.source, relationship.target, relationship.type)
                    if rel_key not in retrieved_relationships:
                        retrieved_relationships.add(rel_key)
                        relationships_list.append(relationship)

        logger.info(
            "Retrieval complete. Found %d entities and %d relationships",
            len(top_entities), len(relationships_list)
        )

        return RetrievalResult(
            query=query,
            entities=top_entities,
            relationships=relationships_list,
            scores={e.id: combined_scores.get(e.id, 0.0) for e in top_entities}
        )

    def _get_local_subgraph(
        self,
        seed_ids: List[str],
        depth: int
    ) -> Tuple[Set[str], Dict[str, List[Tuple[str, float]]]]:
        """Finds the local neighborhood of seed nodes up to `depth` hops.

        Returns:
            A tuple containing:
            - The set of all node IDs in the local subgraph.
            - An adjacency list mapping node ID -> list of (neighbor_id, edge_weight).
        """
        local_nodes: Set[str] = set(seed_ids)
        adjacency: Dict[str, List[Tuple[str, float]]] = {}

        # Queue for BFS: (node_id, current_depth)
        queue = [(node_id, 0) for node_id in seed_ids]
        visited = set(seed_ids)

        while queue:
            curr_id, curr_depth = queue.pop(0)

            # Initialize adjacency list for this node
            if curr_id not in adjacency:
                adjacency[curr_id] = []

            if curr_depth < depth:
                neighbors = self.graph_store.get_neighbors(curr_id)
                for neighbor_entity, relationship in neighbors:
                    neighbor_id = neighbor_entity.id

                    # Add to local nodes
                    local_nodes.add(neighbor_id)

                    # Record edge weight (default to 1.0 if not present or <= 0)
                    weight = max(0.01, relationship.weight)
                    adjacency[curr_id].append((neighbor_id, weight))

                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append((neighbor_id, curr_depth + 1))

        return local_nodes, adjacency

    def _compute_local_ppr(
        self,
        local_nodes: Set[str],
        adjacency: Dict[str, List[Tuple[str, float]]],
        personalization: Dict[str, float],
        max_iter: int = 20,
        tol: float = 1e-6
    ) -> Dict[str, float]:
        """Computes Personalized PageRank (RWR) on the local subgraph.

        Args:
            local_nodes: Set of node IDs in the local subgraph.
            adjacency: Adjacency list mapping node ID -> list of (neighbor_id, edge_weight).
            personalization: Dict mapping node ID -> personalization value (must sum to 1.0).
            max_iter: Maximum number of power iterations.
            tol: Convergence tolerance.

        Returns:
            Dict mapping node ID -> PPR score.
        """
        if not local_nodes:
            return {}

        # Initialize scores with personalization vector
        scores = {node: personalization.get(node, 0.0) for node in local_nodes}

        # If personalization is empty or all zeros, initialize uniformly
        if sum(personalization.values()) == 0:
            uniform_val = 1.0 / len(local_nodes)
            personalization = {node: uniform_val for node in local_nodes}
            scores = personalization.copy()

        # Precompute out-degree weights for transition probabilities
        out_weights: Dict[str, float] = {}
        for node in local_nodes:
            neighbors = adjacency.get(node, [])
            # Only sum weights of neighbors that are actually in our local subgraph
            total_w = sum(w for n_id, w in neighbors if n_id in local_nodes)
            out_weights[node] = total_w

        # Power iteration
        for _ in range(max_iter):
            next_scores = {node: 0.0 for node in local_nodes}
            dangling_sum = 0.0

            # Distribute scores
            for node in local_nodes:
                node_score = scores[node]
                total_w = out_weights[node]

                if total_w > 0:
                    neighbors = adjacency.get(node, [])
                    for neighbor_id, weight in neighbors:
                        if neighbor_id in local_nodes:
                            # Transition probability is weight / total_w
                            transition_prob = weight / total_w
                            next_scores[neighbor_id] += (1.0 - self.alpha) * node_score * transition_prob
                else:
                    # Dangling node (no outgoing edges in local subgraph)
                    dangling_sum += node_score

            # Add restart (personalization) and distribute dangling node scores
            for node in local_nodes:
                # Dangling nodes distribute their score according to the personalization vector
                next_scores[node] += (1.0 - self.alpha) * dangling_sum * personalization.get(node, 0.0)
                # Restart probability
                next_scores[node] += self.alpha * personalization.get(node, 0.0)

            # Check convergence
            err = sum(abs(next_scores[node] - scores[node]) for node in local_nodes)
            scores = next_scores
            if err < tol:
                break

        return scores
