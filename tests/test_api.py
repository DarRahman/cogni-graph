# tests/test_api.py - Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation and hierarchical entity-relation extraction.
# Contributed by Claude Code

"""Unit tests for the FastAPI application endpoints."""

from fastapi.testclient import TestClient
from cognigraph.api import app


def test_api_status() -> None:
    """Tests the status endpoint of the API."""
    client = TestClient(app)
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert "graph" in data
    assert "vector_store" in data


def test_api_ingest_and_retrieve() -> None:
    """Tests the ingest, retrieve, and consolidate endpoints."""
    client = TestClient(app)

    # Ingest
    ingest_payload = {
        "messages": [
            {
                "role": "user",
                "content": "David works at Apple. David likes Swift.",
                "timestamp": "2024-01-01T00:00:00Z"
            }
        ]
    }
    response = client.post("/ingest", json=ingest_payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["entities"]) >= 3

    # Retrieve
    retrieve_payload = {
        "query": "Where does David work?",
        "k": 2,
        "depth": 1
    }
    response = client.post("/retrieve", json=retrieve_payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["entities"]) > 0

    # Consolidate
    response = client.post("/consolidate")
    assert response.status_code == 200
    assert response.json()["status"] == "success"
