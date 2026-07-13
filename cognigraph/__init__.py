# cognigraph/__init__.py - Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation and hierarchical entity-relation extraction.
# Contributed by Claude Code

"""CogniGraph: A stateful long-term memory engine for LLM agents."""

import logging

# Configure logging for the package
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("cognigraph")
logger.info("Initializing CogniGraph package")

__version__ = "0.1.0"
__all__ = [
    "config",
    "models",
    "extractor",
    "graph_store",
    "neo4j_store",
    "vector_store",
    "retriever",
    "pipeline",
    "episodic_buffer",
]
