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
    assert "episodic_buffer" in data


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


def test_api_episodic_endpoints() -> None:
    """Tests the episodic buffer endpoints (/episodes, /episodes/processed)."""
    client = TestClient(app)

    # Ingest into episodic buffer
    payload = {
        "messages": [
            {
                "role": "user",
                "content": "Eve works at Netflix.",
                "timestamp": "2024-01-01T00:00:00Z",
                "metadata": {"session_id": "session_eve"}
            }
        ]
    }
    response = client.post("/episodes", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "message_ids" in data
    assert len(data["message_ids"]) == 1
    msg_id = data["message_ids"][0]

    # Get episodes (unprocessed)
    response = client.get("/episodes?unprocessed_only=true")
    assert response.status_code == 200
    episodes = response.json()
    assert len(episodes) >= 1
    assert any(ep.get("id") == msg_id for ep in episodes)

    # Consolidate to process the episodes
    response = client.post("/consolidate")
    assert response.status_code == 200

    # Get episodes (unprocessed should be empty now for this session)
    response = client.get("/episodes?session_id=session_eve&unprocessed_only=true")
    assert response.status_code == 200
    assert len(response.json()) == 0

    # Clear processed episodes
    response = client.request("DELETE", "/episodes/processed")
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_api_workflow_consolidation() -> None:
    """Tests the LangGraph workflow consolidation endpoint."""
    client = TestClient(app)

    # Ingest into episodic buffer
    payload = {
        "messages": [
            {
                "role": "user",
                "content": "Frank works at Microsoft.",
                "timestamp": "2024-01-01T00:00:00Z",
                "metadata": {"session_id": "session_frank"}
            }
        ]
    }
    response = client.post("/episodes", json=payload)
    assert response.status_code == 200

    # Run workflow consolidation
    workflow_payload = {
        "session_id": "session_frank",
        "decay_factor": 0.95,
        "pruning_threshold": 0.1,
        "similarity_threshold": 0.85,
        "forgetting_age_days": 30.0
    }
    response = client.post("/consolidate/workflow", json=workflow_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["merged_entities_count"] >= 0
    assert data["pruned_relationships_count"] >= 0
    assert data["forgotten_entities_count"] >= 0
    assert "started_at" in data
    assert "completed_at" in data
