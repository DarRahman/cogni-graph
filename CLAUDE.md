# CLAUDE.md

## Development Commands
* Run API: `python -m cognigraph.api`
* Run CLI: `python -m cognigraph.cli`
* Run tests: `pytest`
* Lint/Format: `ruff check .`

## Codebase Architecture
CogniGraph is a production-grade, long-term memory consolidation engine designed for LLM agents. Instead of relying on naive vector search over raw chat histories, CogniGraph asynchronously processes conversation streams to build a dynamic, semantic knowledge graph. It extracts entities, relationships, and temporal events, resolving duplicates and clustering related concepts. Retrieval queries perform a hybrid search: vector similarity locates starting nodes, and graph traversals retrieve rich contextual subgraphs. This architecture prevents context window bloat, resolves coreference issues across sessions, and enables agents to maintain a coherent, evolving mental model of users and tasks over time.
