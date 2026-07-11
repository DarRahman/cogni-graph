# CLAUDE.md

## Development Commands
* Run application: `python app.py` (or correct command for language)
* Run tests: `pytest`
* Lint/Format: `ruff check .`

## Codebase Architecture
CogniGraph is a production-grade, long-term memory consolidation engine designed for LLM agents. Instead of relying on naive vector search over raw chat histories, CogniGraph asynchronously processes conversation streams to build a dynamic, semantic knowledge graph. It extracts entities, relationships, and temporal events, resolving duplicates and clustering related concepts. Retrieval queries perform a hybrid search: vector similarity locates starting nodes, and graph traversals retrieve rich contextual subgraphs. This architecture prevents context window bloat, resolves coreference issues across sessions, and enables agents to maintain a coherent, evolving mental model of users and tasks over time.

Initial file structure:
* `pyproject.toml`
* `README.md`
* `cognigraph/__init__.py`
* `cognigraph/config.py`
* `cognigraph/models.py`
* `cognigraph/extractor.py`
* `cognigraph/graph_store.py`
* `cognigraph/vector_store.py`
* `cognigraph/retriever.py`
* `cognigraph/pipeline.py`
* `cognigraph/api.py`
* `tests/test_extractor.py`
* `tests/test_graph_store.py`
* `tests/test_retriever.py`
