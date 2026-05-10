"""Shared data contracts and Pydantic models."""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone


class ChatRequest(BaseModel):
    """Request model for chat interactions."""
    message: str = Field(..., min_length=1, description="User message")
    session_id: str = Field(..., description="Session identifier")


class ChatResponse(BaseModel):
    """Response model for chat interactions."""
    response: str = Field(..., description="AI response")
    session_id: str = Field(..., description="Session identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Health status")
    version: Optional[str] = Field(None, description="API version")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MessageHistory(BaseModel):
    """Model for storing conversation history."""
    session_id: str
    messages: List[dict]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# Phase 2: Retrieval Infrastructure Models
# =============================================================================


class Document(BaseModel):
    """Represents a loaded document with text and metadata."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "This is the content of the document...",
                "metadata": {
                    "source": "refund_policy.pdf",
                    "page": 4,
                    "created_at": "2026-05-01",
                    "department": "support"
                }
            }
        }
    )
    
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """Represents a text chunk derived from a document."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Customers may request refunds within 30 days...",
                "metadata": {
                    "source": "refund_policy.pdf",
                    "page": 4,
                    "chunk_index": 2
                },
                "chunk_id": "chunk_abc123",
                "start_index": 150,
                "end_index": 450
            }
        }
    )
    
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk_id: Optional[str] = None
    start_index: Optional[int] = None
    end_index: Optional[int] = None


class EmbeddingResult(BaseModel):
    """Represents an embedding vector result."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Sample text for embedding",
                "embedding": [0.1, -0.2, 0.3],
                "model": "text-embedding-3-large",
                "dimensions": 1536
            }
        }
    )
    
    text: str
    embedding: list[float]
    model: str
    dimensions: int


class RetrievalResult(BaseModel):
    """Represents a retrieval result with score and source info."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Customers may request refunds within 30 days...",
                "score": 0.92,
                "source": "refund_policy.pdf",
                "metadata": {
                    "page": 4,
                    "department": "support"
                },
                "chunk_id": "chunk_abc123"
            }
        }
    )
    
    text: str
    score: float
    source: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk_id: Optional[str] = None


class RetrievalRequest(BaseModel):
    """Request schema for retrieval endpoint."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What is the refund policy?",
                "top_k": 10,
                "filters": {"department": "support"}
            }
        }
    )
    
    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    filters: Optional[Dict[str, Any]] = None


class RetrievalResponse(BaseModel):
    """Response schema for retrieval endpoint."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "results": [
                    {
                        "text": "Customers may request refunds within 30 days...",
                        "score": 0.92,
                        "source": "refund_policy.pdf",
                        "metadata": {},
                        "chunk_id": "chunk_abc123"
                    }
                ],
                "query": "What is the refund policy?",
                "total_results": 1
            }
        }
    )
    
    results: list[RetrievalResult]
    query: str
    total_results: int


class IngestionResult(BaseModel):
    """Result of document ingestion process."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_id": "doc_xyz789",
                "chunks_created": 15,
                "source": "refund_policy.pdf",
                "status": "success",
                "errors": []
            }
        }
    )
    
    document_id: str
    chunks_created: int
    source: str
    status: str
    errors: list[str] = Field(default_factory=list)
