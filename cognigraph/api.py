# cognigraph/api.py - Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation and hierarchical entity-relation extraction.
# Contributed by Claude Code

"""FastAPI application exposing memory read/write endpoints."""

import logging
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from cognigraph.config import settings
from cognigraph.extractor import RuleBasedExtractor
from cognigraph.graph_store import NetworkXGraphStore
from cognigraph.models import ChatMessage, ExtractionResult, RetrievalResult
from cognigraph.pipeline import ConsolidationPipeline, MockEmbedder
from cognigraph.retriever import HybridRetriever
from cognigraph.vector_store import SimpleVectorStore

logger = logging.getLogger("cognigraph.api")

app = FastAPI(
    title="CogniGraph API",
    description="Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation.",
    version="0.1.0"
)

# Initialize components
graph_store = NetworkXGraphStore()
vector_store = SimpleVectorStore()
extractor = RuleBasedExtractor()
embedder = MockEmbedder(dimension=settings.EMBEDDING_DIMENSION)
pipeline = ConsolidationPipeline(graph_store, vector_store, extractor, embedder)
retriever = HybridRetriever(graph_store, vector_store)


class IngestRequest(BaseModel):
    """Request model for ingesting chat messages."""
    messages: List[ChatMessage]


class RetrieveRequest(BaseModel):
    """Request model for retrieving context."""
    query: str
    k: int = 3
    depth: int = 1


@app.on_event("startup")
def startup_event() -> None:
    """Loads stores from disk on startup if files exist."""
    logger.info("Starting up CogniGraph API")
    graph_store.load_from_disk(settings.GRAPH_DB_PATH)
    vector_store.load_from_disk(settings.VECTOR_DB_PATH)


@app.on_event("shutdown")
def shutdown_event() -> None:
    """Saves stores to disk on shutdown."""
    logger.info("Shutting down CogniGraph API")
    graph_store.save_to_disk(settings.GRAPH_DB_PATH)
    vector_store.save_to_disk(settings.VECTOR_DB_PATH)


@app.post("/ingest", response_model=ExtractionResult)
def ingest_messages(request: IngestRequest) -> ExtractionResult:
    """Ingests chat messages, extracts facts, and updates stores."""
    try:
        result = pipeline.ingest_and_process(request.messages)
        return result
    except Exception as e:
        logger.exception("Failed to ingest messages")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/retrieve", response_model=RetrievalResult)
def retrieve_context(request: RetrieveRequest) -> RetrievalResult:
    """Retrieves a contextual subgraph relevant to the query."""
    try:
        query_vector = embedder.embed_text(request.query)
        result = retriever.retrieve(
            query=request.query,
            query_vector=query_vector,
            k=request.k,
            depth=request.depth
        )
        return result
    except Exception as e:
        logger.exception("Failed to retrieve context")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/consolidate")
def trigger_consolidation() -> Dict[str, str]:
    """Manually triggers the memory consolidation loop."""
    try:
        pipeline.consolidate()
        return {"status": "success", "message": "Consolidation completed successfully"}
    except Exception as e:
        logger.exception("Failed to run consolidation")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/status")
def get_status() -> Dict[str, Any]:
    """Returns the status of the memory stores."""
    return {
        "graph": {
            "nodes_count": graph_store.graph.number_of_nodes(),
            "edges_count": graph_store.graph.number_of_edges()
        },
        "vector_store": {
            "vectors_count": len(vector_store.vectors)
        }
    }


def main() -> None:
    """Runs the FastAPI application using Uvicorn."""
    import uvicorn
    logger.info("Starting API server on %s:%d", settings.API_HOST, settings.API_PORT)
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)


if __name__ == "__main__":
    main()
