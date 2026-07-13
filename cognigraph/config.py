# cognigraph/config.py - Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation and hierarchical entity-relation extraction.
# Contributed by Claude Code

"""Configuration management for CogniGraph using Pydantic Settings."""

import logging
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("cognigraph.config")


class Settings(BaseSettings):
    """Application settings for CogniGraph.

    Loads configuration from environment variables with sensible defaults.
    """

    # Logging configuration
    LOG_LEVEL: str = "INFO"

    # Storage paths
    DATA_DIR: str = "./data"
    VECTOR_DB_PATH: str = "./data/vector_store.json"
    GRAPH_DB_PATH: str = "./data/graph_store.json"
    EPISODIC_DB_PATH: str = "./data/episodic_store.json"

    # LLM & Embedding configuration (for future integration, but configured now)
    LLM_MODEL: str = "gpt-4-turbo"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_API_BASE: Optional[str] = None

    # Consolidation parameters
    CONSOLIDATION_INTERVAL_SECONDS: int = 3600
    RECENCY_DECAY_FACTOR: float = 0.95  # Decay factor for relationship weights
    SIMILARITY_THRESHOLD: float = 0.85  # Threshold for entity merging
    PRUNING_THRESHOLD: float = 0.1  # Threshold below which edges are pruned

    # Neo4j configuration
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    NEO4J_DATABASE: str = "neo4j"
    USE_NEO4J: bool = False

    # API configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    model_config = SettingsConfigDict(
        env_prefix="COGNIGRAPH_",
        case_sensitive=True,
        extra="ignore"
    )


# Global settings instance
settings = Settings()
logger.info("Configuration loaded successfully")
