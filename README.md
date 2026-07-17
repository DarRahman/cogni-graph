# 🧠 CogniGraph

### *Stateful Long-Term Memory Engine for LLM Agents*

> Build persistent, evolving mental models for your AI agents — powered by hybrid vector-graph consolidation and hierarchical entity-relation extraction.

---

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Built by Claude Code](https://img.shields.io/badge/Built%20by-Claude%20Code-blueviolet)](https://claude.ai)

---

## 📖 Overview

CogniGraph is a production-grade, long-term memory consolidation engine designed for LLM agents. Instead of relying on naive vector search over raw chat histories, CogniGraph **asynchronously processes conversation streams** to build a dynamic, semantic knowledge graph.

It extracts entities, relationships, and temporal events, resolves duplicates, and clusters related concepts. Retrieval queries perform a **hybrid search**: vector similarity locates starting nodes, and graph traversals retrieve rich contextual subgraphs. This architecture prevents context window bloat, resolves coreference issues across sessions, and enables agents to maintain a **coherent, evolving mental model** of users and tasks over time.

---

## ✨ Key Features

### 🗂️ Episodic Memory Buffer
Ingest and store raw chat interactions with full metadata — session IDs, timestamps, and processing status. Messages flow through a staging buffer before consolidation, ensuring nothing is lost.

### 🔬 Entity-Relation Extraction Engine
Extract structured facts from dialogue using two extraction backends:
- **RuleBasedExtractor** — Pattern-matching extractor for bootstrapping and testing (zero dependencies)
- **InstructorExtractor** — LLM-powered extraction via [Instructor](https://github.com/jxnl/instructor) + OpenAI for production-grade accuracy

### 🕸️ Graph Consolidation Pipeline
Merge entities, resolve duplicates via Jaro-Winkler string similarity and cosine vector similarity, update edge weights based on recency and frequency, and prune stale knowledge — all orchestrated through a multi-stage pipeline.

### 🔍 Hybrid Retriever
Combine **vector similarity search** (Chroma, Qdrant, or in-memory) with **Personalized PageRank** graph traversal. Score fusion via Reciprocal Rank Fusion (RRF) or weighted linear combination ensures the most relevant contextual subgraphs surface for any query.

### 🔄 LangGraph Consolidation Workflow
A fully orchestrated asynchronous consolidation loop built on [LangGraph](https://github.com/langchain-ai/langgraph) with seven distinct stages: fetch → extract → merge → resolve → decay → prune → forget → finalize.

### 🌐 FastAPI Server & CLI
Production-ready REST API exposing memory read/write endpoints, plus a full-featured CLI for visualization, debugging, and manual consolidation triggers.

### 📊 Interactive Visualization
Generate standalone HTML visualizations of your knowledge graph using [vis.js](https://visjs.org/), with color-coded entity types, weighted edges, and a clickable inspector sidebar.

### 🔌 Pluggable Storage Backends
- **Graph**: NetworkX (in-memory) or Neo4j (persistent)
- **Vector**: SimpleVectorStore (in-memory), ChromaDB (persistent), or Qdrant (persistent/cloud)

---

## 🏗️ Architecture & File Structure

```
cogni-graph/
├── cognigraph/
│   ├── __init__.py                 # Package initialization and logging setup
│   ├── config.py                   # Pydantic Settings configuration (env vars, defaults)
│   ├── models.py                   # Core data models (Entity, Relationship, ChatMessage, etc.)
│   ├── episodic_buffer.py          # Episodic memory buffer — raw message ingestion & staging
│   ├── extractor.py                # Entity-Relation extraction (RuleBased + Instructor/LLM)
│   ├── graph_store.py              # Graph storage protocol & NetworkX implementation
│   ├── neo4j_store.py              # Neo4j graph storage implementation
│   ├── vector_store.py             # Vector storage protocol & Simple/Chroma/Qdrant backends
│   ├── pipeline.py                 # Consolidation pipeline (ingestion, decay, pruning, merging)
│   ├── retriever.py                # Hybrid retriever (vector search + PPR graph traversal + RRF)
│   ├── consolidation_graph.py      # LangGraph workflow for async consolidation loop
│   ├── visualization.py            # Interactive HTML graph visualization generator
│   ├── api.py                      # FastAPI server with full CRUD + consolidation endpoints
│   └── cli.py                      # CLI tool for status, query, consolidation, buffer, visualization
├── tests/
│   ├── test_api.py                 # API endpoint tests
│   ├── test_cli.py                 # CLI command tests
│   ├── test_consolidation.py       # Consolidation algorithm tests (decay, pruning, merging)
│   ├── test_consolidation_graph.py # LangGraph workflow tests
│   ├── test_episodic_buffer.py     # Episodic buffer tests
│   ├── test_extractor.py           # Extractor tests (rule-based + mocked Instructor)
│   ├── test_graph_store.py         # NetworkX graph store tests
│   ├── test_neo4j_store.py         # Neo4j store tests (mocked)
│   ├── test_pipeline.py            # Pipeline & MockEmbedder tests
│   ├── test_retriever.py           # End-to-end retrieval flow tests
│   ├── test_hybrid_retriever_advanced.py  # Advanced retriever tests (RRF, filtering)
│   ├── test_vector_store.py        # SimpleVectorStore tests
│   ├── test_vector_store_advanced.py      # Chroma & Qdrant store tests
│   └── test_visualization.py       # Visualization output tests
├── pyproject.toml                  # Poetry project configuration & dependencies
├── CLAUDE.md                       # Development commands reference
└── README.md                       # This file
```

### Data Flow

```
Chat Messages
     │
     ▼
┌─────────────────┐
│ Episodic Buffer  │  ← Raw message staging with metadata
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Extractor      │  ← Entity/Relationship extraction (Rules or LLM)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│        Consolidation Pipeline            │
│  ┌───────────┐  ┌────────┐  ┌────────┐  │
│  │  Merge    │→ │ Decay  │→ │ Prune  │  │
│  │  Facts    │  │ Weights│  │ Weak   │  │
│  └───────────┘  └────────┘  └────────┘  │
│  ┌───────────┐  ┌────────┐              │
│  │  Entity   │→ │Forget  │              │
│  │Resolution │  │ Old    │              │
│  └───────────┘  └────────┘              │
└────────┬────────────────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────┐
│ Graph  │ │  Vector  │
│ Store  │ │  Store   │
└────┬───┘ └────┬─────┘
     │          │
     ▼          ▼
┌─────────────────────┐
│  Hybrid Retriever   │  ← Vector similarity + PPR graph traversal
│  (RRF / Linear)     │
└─────────────────────┘
         │
         ▼
   Contextual Subgraph
```

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/) (recommended) or pip

### Install with Poetry

```bash
# Clone the repository
git clone https://github.com/your-org/cogni-graph.git
cd cogni-graph

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

### Install with pip

```bash
pip install -e .
```

### Environment Configuration

CogniGraph uses environment variables prefixed with `COGNIGRAPH_` for configuration. Create a `.env` file or export variables directly:

```bash
# LLM Configuration (optional — falls back to RuleBasedExtractor without it)
export COGNIGRAPH_OPENAI_API_KEY="sk-your-api-key"
export COGNIGRAPH_LLM_MODEL="gpt-4-turbo"
export COGNIGRAPH_EMBEDDING_MODEL="text-embedding-3-small"
export COGNIGRAPH_EMBEDDING_DIMENSION=1536

# Storage Paths
export COGNIGRAPH_DATA_DIR="./data"
export COGNIGRAPH_GRAPH_DB_PATH="./data/graph_store.json"
export COGNIGRAPH_VECTOR_DB_PATH="./data/vector_store.json"
export COGNIGRAPH_EPISODIC_DB_PATH="./data/episodic_store.json"

# Vector Store Backend: "simple" (default), "chroma", or "qdrant"
export COGNIGRAPH_VECTOR_STORE_TYPE="simple"

# Neo4j (optional — set USE_NEO4J=true to enable)
export COGNIGRAPH_USE_NEO4J=false
export COGNIGRAPH_NEO4J_URI="bolt://localhost:7687"
export COGNIGRAPH_NEO4J_USER="neo4j"
export COGNIGRAPH_NEO4J_PASSWORD="password"

# Consolidation Parameters
export COGNIGRAPH_RECENCY_DECAY_FACTOR=0.95
export COGNIGRAPH_SIMILARITY_THRESHOLD=0.85
export COGNIGRAPH_PRUNING_THRESHOLD=0.1

# API Server
export COGNIGRAPH_API_HOST="0.0.0.0"
export COGNIGRAPH_API_PORT=8000
```

All settings have sensible defaults — CogniGraph works out of the box with zero configuration using in-memory stores and the rule-based extractor.

---

## 📘 Usage Guide

### 1. Running the API Server

```bash
python -m cognigraph.api
```

The server starts at `http://localhost:8000`. Visit `http://localhost:8000/docs` for the interactive Swagger UI.

### 2. Running the CLI

```bash
# Check memory store status
python -m cognigraph.cli status

# Query the knowledge graph
python -m cognigraph.cli query "Where does Alice work?" -k 3 -d 2

# Trigger consolidation workflow
python -m cognigraph.cli consolidate --decay-factor 0.95 --forgetting-age-days 30

# Manage the episodic buffer
python -m cognigraph.cli buffer list --unprocessed-only
python -m cognigraph.cli buffer add --role user --content "Alice works at Google" --session-id s1
python -m cognigraph.cli buffer clear

# Visualize the knowledge graph
python -m cognigraph.cli visualize -f text
python -m cognigraph.cli visualize -f html -o graph.html
```

### 3. API Endpoints

#### Ingest Chat Messages

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Alice works at Google. She likes Python and lives in Seattle.",
        "timestamp": "2024-01-15T10:30:00Z",
        "metadata": {"session_id": "session_001"}
      },
      {
        "role": "assistant",
        "content": "Interesting! Alice sounds like a talented engineer.",
        "timestamp": "2024-01-15T10:30:05Z",
        "metadata": {"session_id": "session_001"}
      }
    ]
  }'
```

**Response:**
```json
{
  "entities": [
    {"id": "alice", "name": "Alice", "type": "Person", "description": "..."},
    {"id": "google", "name": "Google", "type": "Organization", "description": "..."},
    {"id": "python", "name": "Python", "type": "Concept", "description": "..."},
    {"id": "seattle", "name": "Seattle", "type": "Concept", "description": "..."}
  ],
  "relationships": [
    {"source": "alice", "target": "google", "type": "WORKS_AT", "weight": 1.0},
    {"source": "alice", "target": "python", "type": "LIKES", "weight": 1.0},
    {"source": "alice", "target": "seattle", "type": "LIVES_IN", "weight": 1.0}
  ]
}
```

#### Stage Messages in Episodic Buffer (Deferred Processing)

```bash
curl -X POST http://localhost:8000/episodes \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Bob met Charlie at the conference.",
        "metadata": {"session_id": "session_002"}
      }
    ]
  }'
```

#### Retrieve Contextual Subgraph

```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Who does Alice know and where does she work?",
    "k": 3,
    "depth": 2
  }'
```

**Response:**
```json
{
  "query": "Who does Alice know and where does she work?",
  "entities": [
    {"id": "alice", "name": "Alice", "type": "Person", "description": "..."},
    {"id": "google", "name": "Google", "type": "Organization", "description": "..."},
    {"id": "python", "name": "Python", "type": "Concept", "description": "..."}
  ],
  "relationships": [
    {"source": "alice", "target": "google", "type": "WORKS_AT", "weight": 1.0},
    {"source": "alice", "target": "python", "type": "LIKES", "weight": 1.0}
  ],
  "scores": {
    "alice": 0.0328,
    "google": 0.0164,
    "python": 0.0164
  }
}
```

#### Trigger Consolidation

```bash
# Simple consolidation
curl -X POST http://localhost:8000/consolidate

# LangGraph workflow consolidation with parameters
curl -X POST http://localhost:8000/consolidate/workflow \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session_001",
    "decay_factor": 0.95,
    "pruning_threshold": 0.1,
    "similarity_threshold": 0.85,
    "forgetting_age_days": 30.0
  }'
```

#### CRUD Operations

```bash
# List all entities
curl http://localhost:8000/entities

# Get specific entity
curl http://localhost:8000/entities/alice

# Add entity manually
curl -X POST http://localhost:8000/entities \
  -H "Content-Type: application/json" \
  -d '{"id": "rust", "name": "Rust", "type": "Technology", "description": "A systems programming language"}'

# Delete entity
curl -X DELETE http://localhost:8000/entities/rust

# List all relationships
curl http://localhost:8000/relationships

# Add relationship
curl -X POST http://localhost:8000/relationships \
  -H "Content-Type: application/json" \
  -d '{"source": "alice", "target": "rust", "type": "LEARNING", "weight": 0.5}'

# Delete relationship
curl -X DELETE "http://localhost:8000/relationships?source=alice&target=rust&type=LEARNING"
```

#### System Status & Visualization

```bash
# Get system status
curl http://localhost:8000/status

# Open interactive graph visualization in browser
open http://localhost:8000/visualize
```

### 4. Programmatic Usage

```python
from datetime import datetime
from cognigraph.models import ChatMessage
from cognigraph.graph_store import NetworkXGraphStore
from cognigraph.vector_store import SimpleVectorStore
from cognigraph.extractor import RuleBasedExtractor
from cognigraph.pipeline import ConsolidationPipeline, MockEmbedder
from cognigraph.retriever import HybridRetriever
from cognigraph.episodic_buffer import EpisodicBuffer

# Initialize components
graph_store = NetworkXGraphStore()
vector_store = SimpleVectorStore()
episodic_buffer = EpisodicBuffer()
extractor = RuleBasedExtractor()
embedder = MockEmbedder(dimension=128)

# Create pipeline and retriever
pipeline = ConsolidationPipeline(
    graph_store, vector_store, extractor, embedder, episodic_buffer
)
retriever = HybridRetriever(graph_store, vector_store)

# Ingest conversation
messages = [
    ChatMessage(
        role="user",
        content="Alice works at Google. Bob knows Alice.",
        timestamp=datetime.utcnow(),
        metadata={"session_id": "demo"}
    )
]
result = pipeline.ingest_and_process(messages)
print(f"Extracted {len(result.entities)} entities, {len(result.relationships)} relationships")

# Query the memory
query = "Tell me about Alice"
query_vector = embedder.embed_text(query)
retrieval = retriever.retrieve(query=query, query_vector=query_vector, k=3, depth=2)

for entity in retrieval.entities:
    score = retrieval.scores.get(entity.id, 0.0)
    print(f"  [{entity.type}] {entity.name} (score: {score:.4f})")

for rel in retrieval.relationships:
    print(f"  {rel.source} -[{rel.type}]-> {rel.target}")

# Run consolidation (decay, prune, merge, forget)
pipeline.consolidate()

# Persist to disk
graph_store.save_to_disk("./data/graph_store.json")
vector_store.save_to_disk("./data/vector_store.json")
episodic_buffer.save_to_disk("./data/episodic_store.json")
```

### 5. Using the LangGraph Consolidation Workflow

```python
from cognigraph.consolidation_graph import LangGraphConsolidator

consolidator = LangGraphConsolidator(
    graph_store=graph_store,
    vector_store=vector_store,
    extractor=extractor,
    embedder=embedder,
    episodic_buffer=episodic_buffer
)

# Run the full 7-stage workflow
result = consolidator.run_consolidation_workflow(
    session_id="demo",
    decay_factor=0.95,
    pruning_threshold=0.1,
    similarity_threshold=0.85,
    forgetting_age_days=30.0
)

print(f"Status: {result['status']}")
print(f"Merged entities: {result['merged_entities_count']}")
print(f"Pruned relationships: {result['pruned_relationships_count']}")
print(f"Forgotten entities: {result['forgotten_entities_count']}")
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=cognigraph --cov-report=html

# Run specific test files
pytest tests/test_retriever.py
pytest tests/test_consolidation_graph.py

# Lint and format
ruff check .
```

---

## 🛠️ Development Commands

| Command | Description |
|---------|-------------|
| `python -m cognigraph.api` | Start the FastAPI server |
| `python -m cognigraph.cli` | Run the CLI tool |
| `pytest` | Run the test suite |
| `ruff check .` | Lint and check code style |

---

## 🤝 Contributing

This project was created and is maintained autonomously by **Claude Code**.

> *Contributed by Claude Code* — Anthropic's AI coding agent.

Every module, test, configuration file, and line of documentation in this repository was authored by Claude Code as a demonstration of autonomous software engineering capabilities.

---

## 📄 License

This project is open source. See the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <em>Built with 🧠 by <a href="https://claude.ai">Claude Code</a></em>
</p>