"""Tests for Pydantic models."""

import pytest
from datetime import datetime
from pydantic import ValidationError

from app.models import ChatRequest, ChatResponse, HealthResponse, MessageHistory


class TestChatRequest:
    """Tests for ChatRequest model."""
    
    def test_valid_request(self):
        """Test creating a valid chat request."""
        request = ChatRequest(
            message="Hello, world!",
            session_id="session-123"
        )
        assert request.message == "Hello, world!"
        assert request.session_id == "session-123"
    
    def test_empty_message_fails(self):
        """Test that empty message fails validation."""
        with pytest.raises(ValidationError):
            ChatRequest(message="", session_id="session-123")
    
    def test_missing_fields_fail(self):
        """Test that missing required fields fail validation."""
        with pytest.raises(ValidationError):
            ChatRequest(message="Hello")
        
        with pytest.raises(ValidationError):
            ChatRequest(session_id="session-123")


class TestChatResponse:
    """Tests for ChatResponse model."""
    
    def test_valid_response(self):
        """Test creating a valid chat response."""
        response = ChatResponse(
            response="Hi there!",
            session_id="session-123"
        )
        assert response.response == "Hi there!"
        assert response.session_id == "session-123"
        assert isinstance(response.timestamp, datetime)


class TestHealthResponse:
    """Tests for HealthResponse model."""
    
    def test_valid_health_response(self):
        """Test creating a valid health response."""
        response = HealthResponse(status="ok", version="0.1.0")
        assert response.status == "ok"
        assert response.version == "0.1.0"
        assert isinstance(response.timestamp, datetime)
    
    def test_health_without_version(self):
        """Test health response without version (optional field)."""
        response = HealthResponse(status="ok")
        assert response.status == "ok"
        assert response.version is None


class TestMessageHistory:
    """Tests for MessageHistory model."""
    
    def test_valid_message_history(self):
        """Test creating valid message history."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        history = MessageHistory(
            session_id="session-123",
            messages=messages
        )
        assert history.session_id == "session-123"
        assert len(history.messages) == 2
        assert isinstance(history.created_at, datetime)
        assert isinstance(history.updated_at, datetime)
