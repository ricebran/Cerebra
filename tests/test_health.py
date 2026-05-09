"""Tests for health endpoint."""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test that the health endpoint returns a successful response."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_root_endpoint():
    """Test that the root endpoint returns API information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert data["version"] == "0.1.0"


def test_chat_endpoint():
    """Test that the chat endpoint accepts valid requests."""
    payload = {
        "message": "Hello, world!",
        "session_id": "test-session-123"
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert data["session_id"] == payload["session_id"]


def test_chat_endpoint_validation():
    """Test that the chat endpoint validates required fields."""
    # Missing message
    payload = {"session_id": "test-session-123"}
    response = client.post("/chat", json=payload)
    assert response.status_code == 422
    
    # Missing session_id
    payload = {"message": "Hello"}
    response = client.post("/chat", json=payload)
    assert response.status_code == 422
    
    # Empty message
    payload = {"message": "", "session_id": "test-session-123"}
    response = client.post("/chat", json=payload)
    assert response.status_code == 422
