# cognigraph

> Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation and hierarchical entity-relation extraction.

## Overview
CogniGraph is a production-grade, long-term memory consolidation engine designed for LLM agents. Instead of relying on naive vector search over raw chat histories, CogniGraph asynchronously processes conversation streams to build a dynamic, semantic knowledge graph. It extracts entities, relationships, and temporal events, resolving duplicates and clustering related concepts. Retrieval queries perform a hybrid search: vector similarity locates starting nodes, and graph traversals retrieve rich contextual subgraphs. This architecture prevents context window bloat, resolves coreference issues across sessions, and enables agents to maintain a coherent, evolving mental model of users and tasks over time.

## Backlog
- [x] Setup project scaffolding with Poetry, strict typing (mypy), and unit testing suite (pytest).
- [x] Implement the Episodic Memory Buffer to ingest and store raw chat interactions with metadata.
- [x] Build the Entity-Relation Extraction Engine using LangChain/Instructor to extract structured facts from dialogue.
- [x] Develop the Graph Consolidation Pipeline using NetworkX/Neo4j to merge entities, resolve duplicates, and update edge weights based on recency and frequency.
- [x] Create the Hybrid Retriever combining vector similarity search (Chroma/Qdrant) and graph traversal algorithms.
- [x] Design a LangGraph workflow to orchestrate the asynchronous consolidation loop (compaction, pruning, and forgetting).
- [x] Build a FastAPI server exposing the memory read/write endpoints and a CLI tool for memory visualization and debugging.

---
*Created and maintained autonomously by Claude Code.*
