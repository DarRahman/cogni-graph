# cognigraph/api.py - Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation and hierarchical entity-relation extraction.
# Contributed by Claude Code

"""FastAPI application exposing memory read/write endpoints."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from cognigraph.config import settings
from cognigraph.extractor import InstructorExtractor, RuleBasedExtractor
from cognigraph.graph_store import NetworkXGraphStore
from cognigraph.models import ChatMessage, ExtractionResult, RetrievalResult, StoredMessage
from cognigraph.pipeline import ConsolidationPipeline, MockEmbedder
from cognigraph.retriever import HybridRetriever
from cognigraph.vector_store import SimpleVectorStore
from cognigraph.episodic_buffer import EpisodicBuffer

logger = logging.getLogger("cognigraph.api")

app = FastAPI(
    title="CogniGraph API",
    description="Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation.",
    version="0.1.0"
)

# Initialize components
graph_store = NetworkXGraphStore()
vector_store = SimpleVectorStore()
episodic_buffer = EpisodicBuffer()

# Dynamically choose extractor based on configuration
if settings.OPENAI_API_KEY:
    logger.info("Initializing InstructorExtractor for API")
    extractor = InstructorExtractor()
else:
    logger.warning("COGNIGRAPH_OPENAI_API_KEY not set. Falling back to RuleBasedExtractor.")
    extractor = RuleBasedExtractor()

embedder = MockEmbedder(dimension=settings.EMBEDDING_DIMENSION)
pipeline = ConsolidationPipeline(graph_store, vector_store, extractor, embedder, episodic_buffer)
retriever = HybridRetriever(graph_store, vector_store)


class IngestRequest(BaseModel):
    """Request model for ingesting chat messages."""
    messages: List[ChatMessage]


class IngestEpisodesResponse(BaseModel):
    """Request model for ingesting raw chat messages."""
    message_ids: List[str]


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
    episodic_buffer.load_from_disk(settings.EPISODIC_DB_PATH)


@app.on_event("shutdown")
def shutdown_event() -> None:
    """Saves stores to disk on shutdown."""
    logger.info("Shutting down CogniGraph API")
    graph_store.save_to_disk(settings.GRAPH_DB_PATH)
    vector_store.save_to_disk(settings.VECTOR_DB_PATH)
    episodic_buffer.save_to_disk(settings.EPISODIC_DB_PATH)


@app.post("/ingest", response_model=ExtractionResult)
def ingest_messages(request: IngestRequest) -> ExtractionResult:
    """Ingests chat messages, extracts facts, and updates stores."""
    try:
        result = pipeline.ingest_and_process(request.messages)
        return result
    except Exception as e:
        logger.exception("Failed to ingest messages")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/episodes", response_model=IngestEpisodesResponse)
def ingest_episodes(request: IngestRequest) -> IngestEpisodesResponse:
    """Ingests raw chat messages into the episodic buffer without immediate consolidation."""
    try:
        message_ids = episodic_buffer.add_messages(request.messages)
        return IngestEpisodesResponse(message_ids=message_ids)
    except Exception as e:
        logger.exception("Failed to ingest episodes")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/episodes", response_model=List[StoredMessage])
def get_episodes(
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    unprocessed_only: bool = Query(False, description="Only return unprocessed messages"),
    limit: Optional[int] = Query(None, description="Limit the number of returned messages")
) -> List[StoredMessage]:
    """Retrieves stored messages from the episodic buffer."""
    try:
        return episodic_buffer.get_messages(
            session_id=session_id,
            unprocessed_only=unprocessed_only,
            limit=limit
        )
    except Exception as e:
        logger.exception("Failed to retrieve episodes")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.delete("/episodes/processed")
def clear_processed_episodes(
    before_timestamp: Optional[datetime] = Query(None, description="Only clear messages processed before this timestamp")
) -> Dict[str, Any]:
    """Clears processed messages from the episodic buffer."""
    try:
        deleted_count = episodic_buffer.clear_processed(before_timestamp=before_timestamp)
        return {"status": "success", "deleted_count": deleted_count}
    except Exception as e:
        logger.exception("Failed to clear processed episodes")
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
    unprocessed_count = len(episodic_buffer.get_messages(unprocessed_only=True))
    return {
        "graph": {
            "nodes_count": graph_store.graph.number_of_nodes(),
            "edges_count": graph_store.graph.number_of_edges()
        },
        "vector_store": {
            "vectors_count": len(vector_store.vectors)
        },
        "episodic_buffer": {
            "total_messages": len(episodic_buffer.messages),
            "unprocessed_messages": unprocessed_count
        }
    }


def main() -> None:
    """Runs the FastAPI application using Uvicorn."""
    import uvicorn
    logger.info("Starting API server on %s:%d", settings.API_HOST, settings.API_PORT)
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)


if __name__ == "__main__":
    main()
