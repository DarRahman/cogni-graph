# cognigraph/cli.py - Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation and hierarchical entity-relation extraction.
# Contributed by Claude Code

"""CLI tool for memory visualization, debugging, and triggering consolidation."""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from cognigraph.config import settings
from cognigraph.graph_store import NetworkXGraphStore, GraphStore
from cognigraph.neo4j_store import Neo4jGraphStore
from cognigraph.vector_store import SimpleVectorStore, VectorStore
from cognigraph.episodic_buffer import EpisodicBuffer
from cognigraph.extractor import Extractor, InstructorExtractor, RuleBasedExtractor
from cognigraph.pipeline import MockEmbedder
from cognigraph.consolidation_graph import LangGraphConsolidator
from cognigraph.retriever import HybridRetriever

logger = logging.getLogger("cognigraph.cli")


def setup_logging(verbose: bool = False) -> None:
    """Configures logging to print clean messages to console.

    Args:
        verbose: If True, sets log level to DEBUG and prints detailed logs.
    """
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    if verbose:
        root_logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    else:
        root_logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(message)s")

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def init_components() -> Tuple[GraphStore, VectorStore, EpisodicBuffer, Extractor, MockEmbedder]:
    """Initializes CogniGraph components based on settings.

    Returns:
        A tuple containing (graph_store, vector_store, episodic_buffer, extractor, embedder).
    """
    graph_store: GraphStore
    if settings.USE_NEO4J:
        graph_store = Neo4jGraphStore(
            uri=settings.NEO4J_URI,
            user=settings.NEO4J_USER,
            password=settings.NEO4J_PASSWORD,
            database=settings.NEO4J_DATABASE
        )
    else:
        graph_store = NetworkXGraphStore()

    vector_store: VectorStore
    if settings.VECTOR_STORE_TYPE == "chroma":
        from cognigraph.vector_store import ChromaVectorStore
        vector_store = ChromaVectorStore(
            path=settings.CHROMA_PATH,
            collection_name=settings.CHROMA_COLLECTION_NAME
        )
    elif settings.VECTOR_STORE_TYPE == "qdrant":
        from cognigraph.vector_store import QdrantVectorStore
        vector_store = QdrantVectorStore(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            collection_name=settings.QDRANT_COLLECTION_NAME,
            dimension=settings.EMBEDDING_DIMENSION
        )
    else:
        vector_store = SimpleVectorStore()

    episodic_buffer = EpisodicBuffer()

    extractor: Extractor
    if settings.OPENAI_API_KEY:
        extractor = InstructorExtractor()
    else:
        extractor = RuleBasedExtractor()

    embedder = MockEmbedder(dimension=settings.EMBEDDING_DIMENSION)

    return graph_store, vector_store, episodic_buffer, extractor, embedder


def load_stores(graph_store: GraphStore, vector_store: VectorStore, episodic_buffer: EpisodicBuffer) -> None:
    """Loads stores from disk.

    Args:
        graph_store: The graph storage instance.
        vector_store: The vector storage instance.
        episodic_buffer: The episodic buffer instance.
    """
    graph_store.load_from_disk(settings.GRAPH_DB_PATH)
    vector_store.load_from_disk(settings.VECTOR_DB_PATH)
    episodic_buffer.load_from_disk(settings.EPISODIC_DB_PATH)


def save_stores(graph_store: GraphStore, vector_store: VectorStore, episodic_buffer: EpisodicBuffer) -> None:
    """Saves stores to disk.

    Args:
        graph_store: The graph storage instance.
        vector_store: The vector storage instance.
        episodic_buffer: The episodic buffer instance.
    """
    graph_store.save_to_disk(settings.GRAPH_DB_PATH)
    vector_store.save_to_disk(settings.VECTOR_DB_PATH)
    episodic_buffer.save_to_disk(settings.EPISODIC_DB_PATH)


def cmd_status(args: argparse.Namespace, graph_store: GraphStore, vector_store: VectorStore, episodic_buffer: EpisodicBuffer) -> None:
    """Handles the 'status' CLI command.

    Args:
        args: Parsed command line arguments.
        graph_store: The graph storage instance.
        vector_store: The vector storage instance.
        episodic_buffer: The episodic buffer instance.
    """
    load_stores(graph_store, vector_store, episodic_buffer)

    if isinstance(graph_store, NetworkXGraphStore):
        nodes_count = graph_store.graph.number_of_nodes()
        edges_count = graph_store.graph.number_of_edges()
    else:
        nodes_count = len(graph_store.get_all_entities())
        edges_count = len(graph_store.get_all_relationships())

    vectors_count = vector_store.count()
    total_messages = len(episodic_buffer.messages)
    unprocessed_messages = len(episodic_buffer.get_messages(unprocessed_only=True))

    logger.info("=== CogniGraph Status ===")
    logger.info(f"Graph Store: {nodes_count} entities, {edges_count} relationships")
    logger.info(f"Vector Store: {vectors_count} vectors")
    logger.info(f"Episodic Buffer: {total_messages} total messages, {unprocessed_messages} unprocessed")


def cmd_query(args: argparse.Namespace, graph_store: GraphStore, vector_store: VectorStore, embedder: MockEmbedder) -> None:
    """Handles the 'query' CLI command.

    Args:
        args: Parsed command line arguments.
        graph_store: The graph storage instance.
        vector_store: The vector storage instance.
        embedder: The embedding generator.
    """
    load_stores(graph_store, vector_store, EpisodicBuffer())

    retriever = HybridRetriever(graph_store, vector_store)
    query_vector = embedder.embed_text(args.query)

    result = retriever.retrieve(
        query=args.query,
        query_vector=query_vector,
        k=args.k,
        depth=args.depth,
        max_nodes=args.max_nodes
    )

    logger.info(f"=== Query Results for: '{args.query}' ===")
    logger.info(f"Retrieved {len(result.entities)} entities and {len(result.relationships)} relationships.\n")

    logger.info("--- Entities ---")
    for entity in result.entities:
        score = result.scores.get(entity.id, 0.0)
        logger.info(f"- [{entity.type}] {entity.name} (ID: {entity.id}, Score: {score:.4f})")
        if entity.description:
            logger.info(f"  Description: {entity.description}")
        if entity.properties:
            logger.info(f"  Properties: {entity.properties}")

    logger.info("\n--- Relationships ---")
    for rel in result.relationships:
        logger.info(f"- {rel.source} -[{rel.type}]-> {rel.target} (Weight: {rel.weight:.4f})")
        if rel.description:
            logger.info(f"  Context: {rel.description}")


def cmd_consolidate(args: argparse.Namespace, graph_store: GraphStore, vector_store: VectorStore, episodic_buffer: EpisodicBuffer, extractor: Extractor, embedder: MockEmbedder) -> None:
    """Handles the 'consolidate' CLI command.

    Args:
        args: Parsed command line arguments.
        graph_store: The graph storage instance.
        vector_store: The vector storage instance.
        episodic_buffer: The episodic buffer instance.
        extractor: The entity-relationship extractor.
        embedder: The embedding generator.
    """
    load_stores(graph_store, vector_store, episodic_buffer)

    consolidator = LangGraphConsolidator(
        graph_store=graph_store,
        vector_store=vector_store,
        extractor=extractor,
        embedder=embedder,
        episodic_buffer=episodic_buffer
    )

    logger.info("Starting LangGraph consolidation workflow...")
    result = consolidator.run_consolidation_workflow(
        session_id=args.session_id,
        decay_factor=args.decay_factor,
        pruning_threshold=args.pruning_threshold,
        similarity_threshold=args.similarity_threshold,
        forgetting_age_days=args.forgetting_age_days
    )

    save_stores(graph_store, vector_store, episodic_buffer)

    logger.info("\n=== Consolidation Workflow Completed ===")
    logger.info(f"Status: {result.get('status')}")
    logger.info(f"Processed Messages: {len(result.get('unprocessed_message_ids', []))}")
    logger.info(f"Merged Entities: {result.get('merged_entities_count', 0)}")
    logger.info(f"Pruned Relationships: {result.get('pruned_relationships_count', 0)}")
    logger.info(f"Forgotten Entities: {result.get('forgotten_entities_count', 0)}")
    if result.get("errors"):
        logger.error(f"Errors encountered: {result.get('errors')}")


def cmd_buffer(args: argparse.Namespace, graph_store: GraphStore, vector_store: VectorStore, episodic_buffer: EpisodicBuffer) -> None:
    """Handles the 'buffer' CLI command.

    Args:
        args: Parsed command line arguments.
        graph_store: The graph storage instance.
        vector_store: The vector storage instance.
        episodic_buffer: The episodic buffer instance.
    """
    load_stores(graph_store, vector_store, episodic_buffer)

    if args.buffer_op == "list":
        messages = episodic_buffer.get_messages(
            session_id=args.session_id,
            unprocessed_only=args.unprocessed_only,
            limit=args.limit
        )
        logger.info(f"=== Episodic Buffer Messages ({len(messages)}) ===")
        for msg in messages:
            status = "UNPROCESSED" if not msg.processed else f"PROCESSED at {msg.processed_at}"
            session = msg.metadata.get("session_id", "N/A")
            logger.info(f"[{msg.timestamp.isoformat()}] ID: {msg.id} | Role: {msg.role} | Session: {session} | Status: {status}")
            logger.info(f"  Content: {msg.content}")

    elif args.buffer_op == "add":
        from cognigraph.models import ChatMessage
        msg = ChatMessage(
            role=args.role,
            content=args.content,
            timestamp=datetime.utcnow(),
            metadata={"session_id": args.session_id} if args.session_id else {}
        )
        msg_id = episodic_buffer.add_message(msg)
        save_stores(graph_store, vector_store, episodic_buffer)
        logger.info(f"Added message to buffer with ID: {msg_id}")

    elif args.buffer_op == "clear":
        deleted_count = episodic_buffer.clear_processed()
        save_stores(graph_store, vector_store, episodic_buffer)
        logger.info(f"Cleared {deleted_count} processed messages from episodic buffer.")


def cmd_visualize(args: argparse.Namespace, graph_store: GraphStore, vector_store: VectorStore, episodic_buffer: EpisodicBuffer) -> None:
    """Handles the 'visualize' CLI command.

    Args:
        args: Parsed command line arguments.
        graph_store: The graph storage instance.
        vector_store: The vector storage instance.
        episodic_buffer: The episodic buffer instance.
    """
    load_stores(graph_store, vector_store, episodic_buffer)

    if getattr(args, "format", "text") == "html":
        from cognigraph.visualization import generate_visual_html
        output_path = getattr(args, "output", "cognigraph_vis.html")
        logger.info("Generating interactive HTML visualization...")
        try:
            html_content = generate_visual_html(graph_store)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            logger.info(f"Visualization saved successfully to: {os.path.abspath(output_path)}")
            logger.info("Open this file in a web browser to view the interactive graph.")
        except Exception as e:
            logger.error(f"Failed to generate HTML visualization: {e}")
        return

    entities = graph_store.get_all_entities()
    relationships = graph_store.get_all_relationships()

    logger.info("=== Knowledge Graph Visualization ===")
    logger.info(f"Total Entities: {len(entities)}")
    logger.info(f"Total Relationships: {len(relationships)}\n")

    # Group entities by type
    by_type: Dict[str, List[Any]] = {}
    for ent in entities:
        by_type.setdefault(ent.type, []).append(ent)

    logger.info("--- Entities by Type ---")
    for ent_type, ents in by_type.items():
        logger.info(f"\nType: {ent_type} ({len(ents)})")
        for ent in ents:
            deg = graph_store.get_degree(ent.id)
            logger.info(f"  * {ent.name} (ID: {ent.id}, Degree: {deg})")
            if ent.description:
                logger.info(f"    Description: {ent.description}")

    logger.info("\n--- Relationships ---")
    for rel in relationships:
        logger.info(f"  * {rel.source} -[{rel.type}]-> {rel.target} (Weight: {rel.weight:.2f})")
        if rel.description:
            logger.info(f"    Context: {rel.description}")


def main() -> None:
    """Main entry point for the CLI tool."""
    parser = argparse.ArgumentParser(
        description="CogniGraph CLI - Memory visualization, debugging, and consolidation tool."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Commands")

    # Status command
    subparsers.add_parser("status", help="Show status of memory stores")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the hybrid retriever")
    query_parser.add_argument("query", type=str, help="The query string")
    query_parser.add_argument("-k", type=int, default=3, help="Number of seed entities to retrieve")
    query_parser.add_argument("-d", "--depth", type=int, default=2, help="Graph traversal depth")
    query_parser.add_argument("-m", "--max-nodes", type=int, default=10, help="Maximum entities to return")

    # Consolidate command
    consolidate_parser = subparsers.add_parser("consolidate", help="Trigger consolidation workflow")
    consolidate_parser.add_argument("--session-id", type=str, help="Filter messages by session ID")
    consolidate_parser.add_argument("--decay-factor", type=float, help="Decay factor override")
    consolidate_parser.add_argument("--pruning-threshold", type=float, help="Pruning threshold override")
    consolidate_parser.add_argument("--similarity-threshold", type=float, help="Similarity threshold override")
    consolidate_parser.add_argument("--forgetting-age-days", type=float, help="Forgetting age threshold override")

    # Buffer command
    buffer_parser = subparsers.add_parser("buffer", help="Manage episodic buffer")
    buffer_subparsers = buffer_parser.add_subparsers(dest="buffer_op", required=True, help="Buffer operations")

    # Buffer list
    buffer_list_parser = buffer_subparsers.add_parser("list", help="List messages in the buffer")
    buffer_list_parser.add_argument("--session-id", type=str, help="Filter by session ID")
    buffer_list_parser.add_argument("--unprocessed-only", action="store_true", help="Only show unprocessed messages")
    buffer_list_parser.add_argument("--limit", type=int, help="Limit the number of messages")

    # Buffer add
    buffer_add_parser = buffer_subparsers.add_parser("add", help="Add a message to the buffer")
    buffer_add_parser.add_argument("--role", type=str, required=True, choices=["user", "assistant", "system"], help="Speaker role")
    buffer_add_parser.add_argument("--content", type=str, required=True, help="Message content")
    buffer_add_parser.add_argument("--session-id", type=str, help="Session ID in metadata")

    # Buffer clear
    buffer_subparsers.add_parser("clear", help="Clear processed messages from buffer")

    # Visualize command
    visualize_parser = subparsers.add_parser("visualize", help="Visualize the knowledge graph")
    visualize_parser.add_argument(
        "-f", "--format", type=str, choices=["text", "html"], default="text",
        help="Visualization format: 'text' (console output) or 'html' (interactive browser visualization)"
    )
    visualize_parser.add_argument(
        "-o", "--output", type=str, default="cognigraph_vis.html",
        help="Output file path for HTML visualization (default: cognigraph_vis.html)"
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    try:
        graph_store, vector_store, episodic_buffer, extractor, embedder = init_components()

        if args.command == "status":
            cmd_status(args, graph_store, vector_store, episodic_buffer)
        elif args.command == "query":
            cmd_query(args, graph_store, vector_store, embedder)
        elif args.command == "consolidate":
            cmd_consolidate(args, graph_store, vector_store, episodic_buffer, extractor, embedder)
        elif args.command == "buffer":
            cmd_buffer(args, graph_store, vector_store, episodic_buffer)
        elif args.command == "visualize":
            cmd_visualize(args, graph_store, vector_store, episodic_buffer)

        close_fn = getattr(graph_store, "close", None)
        if close_fn and callable(close_fn):
            close_fn()
    except Exception as e:
        logger.exception("CLI command failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
