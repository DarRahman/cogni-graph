# cognigraph/vector_store.py - Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation and hierarchical entity-relation extraction.
# Contributed by Claude Code

"""In-memory vector store with cosine similarity search and serialization."""

import json
import logging
import os
from typing import Any, Dict, List, Tuple
import numpy as np

logger = logging.getLogger("cognigraph.vector_store")


class SimpleVectorStore:
    """A simple in-memory vector store using numpy for cosine similarity."""

    def __init__(self) -> None:
        """Initializes the vector store."""
        self.vectors: Dict[str, List[float]] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        logger.info("SimpleVectorStore initialized")

    def add_vector(self, vector_id: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        """Adds a vector with associated metadata to the store.

        Args:
            vector_id: Unique identifier for the vector (e.g., entity ID).
            vector: List of floats representing the embedding.
            metadata: Dictionary containing metadata.
        """
        self.vectors[vector_id] = vector
        self.metadata[vector_id] = metadata
        logger.debug("Added vector for ID: %s", vector_id)

    def similarity_search(self, query_vector: List[float], k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Performs cosine similarity search against stored vectors.

        Args:
            query_vector: The query embedding vector.
            k: Number of top results to return.

        Returns:
            A list of tuples containing (vector_id, similarity_score, metadata).
        """
        if not self.vectors:
            logger.warning("Vector store is empty. Returning no results.")
            return []

        q_vec = np.array(query_vector)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            logger.warning("Query vector norm is zero.")
            return []

        results = []
        for vid, vec in self.vectors.items():
            v_arr = np.array(vec)
            v_norm = np.linalg.norm(v_arr)
            if v_norm == 0:
                continue
            # Cosine similarity
            similarity = float(np.dot(q_vec, v_arr) / (q_norm * v_norm))
            results.append((vid, similarity, self.metadata[vid]))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        top_k = results[:k]
        logger.info("Similarity search returned %d results", len(top_k))
        return top_k

    def save_to_disk(self, path: str) -> None:
        """Saves the vector store to a JSON file.

        Args:
            path: File path to save the store.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "vectors": self.vectors,
            "metadata": self.metadata
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Vector store saved to %s", path)

    def load_from_disk(self, path: str) -> None:
        """Loads the vector store from a JSON file.

        Args:
            path: File path to load the store from.
        """
        if not os.path.exists(path):
            logger.warning("Vector store file %s does not exist. Starting empty.", path)
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vectors = data.get("vectors", {})
        self.metadata = data.get("metadata", {})
        logger.info("Vector store loaded from %s with %d vectors", path, len(self.vectors))
