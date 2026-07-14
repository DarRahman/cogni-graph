# cognigraph/vector_store.py - Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation and hierarchical entity-relation extraction.
# Contributed by Claude Code

"""In-memory vector store with cosine similarity search and serialization."""

import json
import logging
import os
from typing import Any, Dict, List, Tuple, Optional, Protocol, runtime_checkable
import numpy as np

logger = logging.getLogger("cognigraph.vector_store")

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    import qdrant_client
    from qdrant_client.http import models as qmodels
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


@runtime_checkable
class VectorStore(Protocol):
    """Protocol defining the interface for vector storage engines."""

    def add_vector(self, vector_id: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        """Adds a vector with associated metadata to the store.

        Args:
            vector_id: Unique identifier for the vector.
            vector: List of floats representing the embedding.
            metadata: Dictionary containing metadata.
        """
        ...

    def get_vector(self, vector_id: str) -> Optional[List[float]]:
        """Retrieves a vector by its ID.

        Args:
            vector_id: The ID of the vector.

        Returns:
            The vector if found, else None.
        """
        ...

    def delete_vector(self, vector_id: str) -> None:
        """Deletes a vector from the store.

        Args:
            vector_id: The ID of the vector to delete.
        """
        ...

    def similarity_search(self, query_vector: List[float], k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Performs similarity search against stored vectors.

        Args:
            query_vector: The query embedding vector.
            k: Number of top results to return.

        Returns:
            A list of tuples containing (vector_id, similarity_score, metadata).
        """
        ...

    def count(self) -> int:
        """Returns the number of vectors in the store.

        Returns:
            The count of vectors.
        """
        ...

    def save_to_disk(self, path: str) -> None:
        """Saves the vector store to disk (if supported).

        Args:
            path: File path to save the store.
        """
        ...

    def load_from_disk(self, path: str) -> None:
        """Loads the vector store from disk (if supported).

        Args:
            path: File path to load the store from.
        """
        ...


class SimpleVectorStore:
    """A simple in-memory vector store using numpy for cosine similarity."""

    def __init__(self) -> None:
        """Initializes the vector store."""
        self.vectors: Dict[str, List[float]] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        logger.info("SimpleVectorStore initialized")

    def add_vector(self, vector_id: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        """Adds a vector with associated metadata to the store."""
        self.vectors[vector_id] = vector
        self.metadata[vector_id] = metadata
        logger.debug("Added vector for ID: %s", vector_id)

    def get_vector(self, vector_id: str) -> Optional[List[float]]:
        """Retrieves a vector by its ID."""
        return self.vectors.get(vector_id)

    def delete_vector(self, vector_id: str) -> None:
        """Deletes a vector from the store."""
        if vector_id in self.vectors:
            del self.vectors[vector_id]
        if vector_id in self.metadata:
            del self.metadata[vector_id]
        logger.debug("Deleted vector for ID: %s", vector_id)

    def similarity_search(self, query_vector: List[float], k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Performs cosine similarity search against stored vectors."""
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

    def count(self) -> int:
        """Returns the number of vectors in the store."""
        return len(self.vectors)

    def save_to_disk(self, path: str) -> None:
        """Saves the vector store to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "vectors": self.vectors,
            "metadata": self.metadata
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Vector store saved to %s", path)

    def load_from_disk(self, path: str) -> None:
        """Loads the vector store from a JSON file."""
        if not os.path.exists(path):
            logger.warning("Vector store file %s does not exist. Starting empty.", path)
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vectors = data.get("vectors", {})
        self.metadata = data.get("metadata", {})
        logger.info("Vector store loaded from %s with %d vectors", path, len(self.vectors))


class ChromaVectorStore:
    """Vector store backed by Chroma DB."""

    def __init__(self, path: str, collection_name: str = "cognigraph") -> None:
        """Initializes the Chroma client and collection.

        Args:
            path: Path to store Chroma database files.
            collection_name: Name of the collection to use.
        """
        if not CHROMA_AVAILABLE:
            raise ImportError(
                "The 'chromadb' package is required to use ChromaVectorStore. "
                "Install it using 'poetry add chromadb' or 'pip install chromadb'."
            )
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        logger.info("ChromaVectorStore initialized at %s for collection %s", path, collection_name)

    def add_vector(self, vector_id: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        """Adds a vector with associated metadata to Chroma."""
        # Chroma requires metadata values to be str, int, float, or bool.
        # We serialize any nested dicts/lists to JSON strings.
        sanitized_metadata = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                sanitized_metadata[k] = v
            else:
                sanitized_metadata[k] = json.dumps(v)

        self.collection.upsert(
            ids=[vector_id],
            embeddings=[vector],
            metadatas=[sanitized_metadata]
        )
        logger.debug("Added vector to Chroma: %s", vector_id)

    def get_vector(self, vector_id: str) -> Optional[List[float]]:
        """Retrieves a vector by its ID from Chroma."""
        # Query Chroma
        result = self.collection.get(ids=[vector_id], include=["embeddings"])
        if result and result.get("embeddings") is not None and len(result["embeddings"]) > 0:
            embeddings = result["embeddings"]
            # Handle numpy arrays, nested structures, or direct lists
            try:
                import numpy as np
                if isinstance(embeddings, np.ndarray):
                    if len(embeddings.shape) > 1:
                        return embeddings[0].tolist()
                    return embeddings.tolist()
            except ImportError:
                pass
            
            # Direct check if it's a list/sequence
            if isinstance(embeddings, list):
                val = embeddings[0]
                if isinstance(val, list):
                    return val
                try:
                    return [float(x) for x in val]
                except (TypeError, ValueError):
                    pass
                return list(embeddings)
            return list(embeddings[0])
        return None

    def delete_vector(self, vector_id: str) -> None:
        """Deletes a vector from Chroma."""
        self.collection.delete(ids=[vector_id])
        logger.debug("Deleted vector from Chroma: %s", vector_id)

    def similarity_search(self, query_vector: List[float], k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Performs similarity search against Chroma vectors."""
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k
        )

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        search_results = []
        ids = results["ids"][0]
        distances = results["distances"][0] if results["distances"] else [0.0] * len(ids)
        metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)

        for vid, dist, meta in zip(ids, distances, metadatas):
            # Convert distance to similarity score.
            # Chroma returns L2 distance or cosine distance.
            # For cosine distance, similarity = 1.0 - distance.
            similarity = 1.0 - dist if dist <= 1.0 else 1.0 / (1.0 + dist)

            # Deserialize any JSON strings in metadata
            deserialized_meta = {}
            for k, v in meta.items():
                if isinstance(v, str):
                    try:
                        deserialized_meta[k] = json.loads(v)
                    except json.JSONDecodeError:
                        deserialized_meta[k] = v
                else:
                    deserialized_meta[k] = v

            search_results.append((vid, similarity, deserialized_meta))

        logger.info("Chroma similarity search returned %d results", len(search_results))
        return search_results

    def count(self) -> int:
        """Returns the number of vectors in the Chroma collection."""
        return int(self.collection.count())

    def save_to_disk(self, path: str) -> None:
        """No-op for Chroma since it is persistent by default."""
        logger.warning("save_to_disk is a no-op for ChromaVectorStore")

    def load_from_disk(self, path: str) -> None:
        """No-op for Chroma since it is persistent by default."""
        logger.warning("load_from_disk is a no-op for ChromaVectorStore")


class QdrantVectorStore:
    """Vector store backed by Qdrant."""

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        collection_name: str = "cognigraph",
        dimension: int = 1536
    ) -> None:
        """Initializes the Qdrant client and collection.

        Args:
            url: Qdrant connection URL. Use ":memory:" for in-memory.
            api_key: Optional API key.
            collection_name: Name of the collection.
            dimension: Dimension of the embedding vectors.
        """
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "The 'qdrant-client' package is required to use QdrantVectorStore. "
                "Install it using 'poetry add qdrant-client' or 'pip install qdrant-client'."
            )

        if url == ":memory:":
            self.client = qdrant_client.QdrantClient(location=":memory:")
        elif url.startswith("path://") or ("/" in url and not url.startswith("http")):
            path = url.replace("path://", "")
            self.client = qdrant_client.QdrantClient(path=path)
        else:
            self.client = qdrant_client.QdrantClient(url=url, api_key=api_key)

        self.collection_name = collection_name
        self.dimension = dimension

        # Ensure collection exists
        try:
            self.client.get_collection(collection_name=self.collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.dimension,
                    distance=qmodels.Distance.COSINE
                )
            )
        logger.info("QdrantVectorStore initialized for collection %s", collection_name)

    def add_vector(self, vector_id: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        """Adds a vector with associated metadata to Qdrant."""
        import uuid
        # Qdrant requires point ID to be UUID or integer.
        # Generate a deterministic UUID from the string ID.
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, vector_id))

        # Store the original string ID in metadata so we can retrieve it
        payload = metadata.copy()
        payload["_original_id"] = vector_id

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                qmodels.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
            ]
        )
        logger.debug("Added vector to Qdrant: %s (point_id: %s)", vector_id, point_id)

    def get_vector(self, vector_id: str) -> Optional[List[float]]:
        """Retrieves a vector by its ID from Qdrant."""
        import uuid
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, vector_id))
        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_vectors=True
            )
            if points and len(points) > 0:
                vector = points[0].vector
                if isinstance(vector, list):
                    return vector
                if isinstance(vector, dict):
                    return list(vector.values())[0]  # type: ignore[return-value]
            return None
        except Exception:
            logger.exception("Failed to retrieve vector from Qdrant: %s", vector_id)
            return None

    def delete_vector(self, vector_id: str) -> None:
        """Deletes a vector from Qdrant."""
        import uuid
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, vector_id))
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qmodels.PointIdsList(
                points=[point_id]
            )
        )
        logger.debug("Deleted vector from Qdrant: %s", vector_id)

    def similarity_search(self, query_vector: List[float], k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Performs similarity search against Qdrant vectors."""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k
        )

        search_results = []
        for hit in results:
            payload = hit.payload or {}
            vid = payload.pop("_original_id", str(hit.id))
            search_results.append((vid, hit.score, payload))

        logger.info("Qdrant similarity search returned %d results", len(search_results))
        return search_results

    def count(self) -> int:
        """Returns the number of vectors in the Qdrant collection."""
        res = self.client.get_collection(collection_name=self.collection_name)
        return res.points_count or 0

    def save_to_disk(self, path: str) -> None:
        """No-op for Qdrant since it is persistent by default."""
        logger.warning("save_to_disk is a no-op for QdrantVectorStore")

    def load_from_disk(self, path: str) -> None:
        """No-op for Qdrant since it is persistent by default."""
        logger.warning("load_from_disk is a no-op for QdrantVectorStore")
