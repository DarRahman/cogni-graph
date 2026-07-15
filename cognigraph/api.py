# cognigraph/api.py - Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation and hierarchical entity-relation extraction.
# Contributed by Claude Code

"""FastAPI application exposing memory read/write endpoints."""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from cognigraph.config import settings
from cognigraph.extractor import Extractor, InstructorExtractor, RuleBasedExtractor
from cognigraph.graph_store import GraphStore, NetworkXGraphStore
from cognigraph.neo4j_store import Neo4jGraphStore
from cognigraph.models import ChatMessage, Entity, ExtractionResult, Relationship, RetrievalResult, StoredMessage
from cognigraph.pipeline import ConsolidationPipeline, MockEmbedder
from cognigraph.retriever import HybridRetriever
from cognigraph.vector_store import VectorStore, SimpleVectorStore
from cognigraph.episodic_buffer import EpisodicBuffer
from cognigraph.consolidation_graph import LangGraphConsolidator
from cognigraph.visualization import generate_visual_html

logger = logging.getLogger("cognigraph.api")

# Initialize components
graph_store: GraphStore
if settings.USE_NEO4J:
    logger.info("Initializing Neo4jGraphStore for API")
    graph_store = Neo4jGraphStore(
        uri=settings.NEO4J_URI,
        user=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
        database=settings.NEO4J_DATABASE
    )
else:
    logger.info("Initializing NetworkXGraphStore for API")
    graph_store = NetworkXGraphStore()

# Dynamically choose vector store based on configuration
vector_store: VectorStore
if settings.VECTOR_STORE_TYPE == "chroma":
    logger.info("Initializing ChromaVectorStore for API")
    from cognigraph.vector_store import ChromaVectorStore
    vector_store = ChromaVectorStore(
        path=settings.CHROMA_PATH,
        collection_name=settings.CHROMA_COLLECTION_NAME
    )
elif settings.VECTOR_STORE_TYPE == "qdrant":
    logger.info("Initializing QdrantVectorStore for API")
    from cognigraph.vector_store import QdrantVectorStore
    vector_store = QdrantVectorStore(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        dimension=settings.EMBEDDING_DIMENSION
    )
else:
    logger.info("Initializing SimpleVectorStore for API")
    vector_store = SimpleVectorStore()

episodic_buffer = EpisodicBuffer()

# Dynamically choose extractor based on configuration
extractor: Extractor
if settings.OPENAI_API_KEY:
    logger.info("Initializing InstructorExtractor for API")
    extractor = InstructorExtractor()
else:
    logger.warning("COGNIGRAPH_OPENAI_API_KEY not set. Falling back to RuleBasedExtractor.")
    extractor = RuleBasedExtractor()

embedder = MockEmbedder(dimension=settings.EMBEDDING_DIMENSION)
pipeline = ConsolidationPipeline(graph_store, vector_store, extractor, embedder, episodic_buffer)
workflow_consolidator = LangGraphConsolidator(graph_store, vector_store, extractor, embedder, episodic_buffer)
retriever = HybridRetriever(graph_store, vector_store)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application."""
    logger.info("Starting up CogniGraph API")
    graph_store.load_from_disk(settings.GRAPH_DB_PATH)
    vector_store.load_from_disk(settings.VECTOR_DB_PATH)
    episodic_buffer.load_from_disk(settings.EPISODIC_DB_PATH)
    yield
    logger.info("Shutting down CogniGraph API")
    graph_store.save_to_disk(settings.GRAPH_DB_PATH)
    vector_store.save_to_disk(settings.VECTOR_DB_PATH)
    episodic_buffer.save_to_disk(settings.EPISODIC_DB_PATH)
    close_fn = getattr(graph_store, "close", None)
    if close_fn and callable(close_fn):
        close_fn()


app = FastAPI(
    title="CogniGraph API",
    description="Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation.",
    version="0.1.0",
    lifespan=lifespan
)


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


class WorkflowConsolidateRequest(BaseModel):
    """Request model for running the consolidation workflow."""
    session_id: Optional[str] = None
    decay_factor: Optional[float] = None
    pruning_threshold: Optional[float] = None
    similarity_threshold: Optional[float] = None
    forgetting_age_days: Optional[float] = None


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


@app.post("/consolidate/workflow")
def trigger_workflow_consolidation(request: Optional[WorkflowConsolidateRequest] = None) -> Dict[str, Any]:
    """Triggers the memory consolidation loop using the LangGraph workflow."""
    try:
        req = request or WorkflowConsolidateRequest()
        result = workflow_consolidator.run_consolidation_workflow(
            session_id=req.session_id,
            decay_factor=req.decay_factor,
            pruning_threshold=req.pruning_threshold,
            similarity_threshold=req.similarity_threshold,
            forgetting_age_days=req.forgetting_age_days
        )
        # Convert datetime objects to string for JSON serialization
        if "started_at" in result and isinstance(result["started_at"], datetime):
            result["started_at"] = result["started_at"].isoformat()
        if "completed_at" in result and isinstance(result["completed_at"], datetime):
            result["completed_at"] = result["completed_at"].isoformat()
        # Remove non-serializable objects like extracted_result and messages
        result.pop("extracted_result", None)
        result.pop("messages", None)
        return result
    except Exception as e:
        logger.exception("Failed to run consolidation workflow")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/status")
def get_status() -> Dict[str, Any]:
    """Returns the status of the memory stores."""
    unprocessed_count = len(episodic_buffer.get_messages(unprocessed_only=True))
    # Safely get node/edge count depending on store type
    if isinstance(graph_store, NetworkXGraphStore):
        nodes_count = graph_store.graph.number_of_nodes()
        edges_count = graph_store.graph.number_of_edges()
    else:
        nodes_count = len(graph_store.get_all_entities())
        edges_count = len(graph_store.get_all_relationships())
        
    return {
        "graph": {
            "nodes_count": nodes_count,
            "edges_count": edges_count
        },
        "vector_store": {
            "vectors_count": vector_store.count()
        },
        "episodic_buffer": {
            "total_messages": len(episodic_buffer.messages),
            "unprocessed_messages": unprocessed_count
        }
    }


@app.get("/visualize", response_class=HTMLResponse)
def get_visualization() -> HTMLResponse:
    """Returns an interactive HTML visualization of the knowledge graph."""
    try:
        html_content = generate_visual_html(graph_store)
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        logger.exception("Failed to generate visualization")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/entities", response_model=List[Entity])
def get_entities() -> List[Entity]:
    """Retrieves all entities from the graph store."""
    try:
        return graph_store.get_all_entities()
    except Exception as e:
        logger.exception("Failed to retrieve entities")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/entities/{entity_id}", response_model=Entity)
def get_entity(entity_id: str) -> Entity:
    """Retrieves a specific entity by ID."""
    try:
        entity = graph_store.get_entity(entity_id)
        if not entity:
            raise HTTPException(status_code=404, detail=f"Entity '{entity_id}' not found")
        return entity
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to retrieve entity")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/entities", response_model=Entity)
def add_entity(entity: Entity) -> Entity:
    """Adds or updates an entity in the graph and vector stores."""
    try:
        graph_store.add_entity(entity)
        # Generate embedding and add to vector store
        embedding_text = f"{entity.name}: {entity.description}"
        vector = embedder.embed_text(embedding_text)
        vector_store.add_vector(
            vector_id=entity.id,
            vector=vector,
            metadata={"name": entity.name, "type": entity.type}
        )
        return entity
    except Exception as e:
        logger.exception("Failed to add entity")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.delete("/entities/{entity_id}")
def delete_entity(entity_id: str) -> Dict[str, str]:
    """Deletes an entity and its connected relationships from the graph and vector stores."""
    try:
        entity = graph_store.get_entity(entity_id)
        if not entity:
            raise HTTPException(status_code=404, detail=f"Entity '{entity_id}' not found")
        
        # Remove all relationships connected to it
        neighbors = graph_store.get_neighbors(entity_id)
        for _, rel in neighbors:
            graph_store.remove_relationship(rel.source, rel.target, rel.type)
            
        graph_store.remove_entity(entity_id)
        vector_store.delete_vector(entity_id)
        return {"status": "success", "message": f"Entity '{entity_id}' deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to delete entity")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/relationships", response_model=List[Relationship])
def get_relationships() -> List[Relationship]:
    """Retrieves all relationships from the graph store."""
    try:
        return graph_store.get_all_relationships()
    except Exception as e:
        logger.exception("Failed to retrieve relationships")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/relationships", response_model=Relationship)
def add_relationship(relationship: Relationship) -> Relationship:
    """Adds or updates a relationship in the graph store."""
    try:
        # Ensure source and target entities exist
        if not graph_store.get_entity(relationship.source):
            graph_store.add_entity(Entity(id=relationship.source, name=relationship.source.capitalize(), type="Concept"))
        if not graph_store.get_entity(relationship.target):
            graph_store.add_entity(Entity(id=relationship.target, name=relationship.target.capitalize(), type="Concept"))
            
        graph_store.add_relationship(relationship)
        return relationship
    except Exception as e:
        logger.exception("Failed to add relationship")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.delete("/relationships")
def delete_relationship(
    source: str = Query(..., description="Source entity ID"),
    target: str = Query(..., description="Target entity ID"),
    type: str = Query(..., description="Relationship type")
) -> Dict[str, str]:
    """Deletes a specific relationship from the graph store."""
    try:
        graph_store.remove_relationship(source, target, type)
        return {"status": "success", "message": f"Relationship '{source} -[{type}]-> {target}' deleted successfully"}
    except Exception as e:
        logger.exception("Failed to delete relationship")
        raise HTTPException(status_code=500, detail=str(e)) from e


def main() -> None:
    """Runs the FastAPI application using Uvicorn."""
    import uvicorn
    logger.info("Starting API server on %s:%d", settings.API_HOST, settings.API_PORT)
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)


if __name__ == "__main__":
    main()
