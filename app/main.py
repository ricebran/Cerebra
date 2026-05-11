"""FastAPI application entrypoint."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time
from typing import List, Optional, Dict, Any

from app.config import API_HOST, API_PORT, DEBUG_MODE, ALLOWED_ORIGINS
from app.models import (
    ChatRequest, ChatResponse, HealthResponse,
    RetrievalRequest, RetrievalResponse, IngestionResult
)
from app.services.ingestion_service import IngestionService


# Global ingestion service instance (initialized on startup)
ingestion_service: Optional[IngestionService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown events."""
    global ingestion_service
    
    # Startup
    print("Starting up the application...")
    
    # Initialize retrieval infrastructure
    try:
        ingestion_service = IngestionService.create_default()
        print("Retrieval infrastructure initialized successfully")
    except Exception as e:
        print(f"Warning: Failed to initialize retrieval infrastructure: {e}")
        ingestion_service = None
    
    yield
    
    # Shutdown
    print("Shutting down the application...")
    # Cleanup connections here in future phases


app = FastAPI(
    title="RAG Chat Application",
    description="A RAG-powered chat application with conversation management",
    version="0.1.0",
    lifespan=lifespan
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok", version="0.1.0")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for user interactions.
    
    This is a placeholder that will be connected to RAG/Agents in Phase 3.
    """
    # Placeholder response - actual AI logic comes in Phase 3
    return ChatResponse(
        response=f"Received your message: {request.message}",
        session_id=request.session_id
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to the RAG Chat API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
        "retrieve": "/retrieve",
        "ingest": "/ingest"
    }


# =============================================================================
# Phase 2: Retrieval Infrastructure Endpoints
# =============================================================================


@app.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(request: RetrievalRequest):
    """
    Retrieve relevant documents using hybrid search.
    
    Combines dense (semantic) and sparse (keyword) retrieval for
    optimal recall and precision.
    
    Args:
        request: RetrievalRequest with query and optional filters
        
    Returns:
        RetrievalResponse with ranked results
    """
    if ingestion_service is None:
        raise HTTPException(
            status_code=503,
            detail="Retrieval infrastructure not initialized"
        )
    
    try:
        # Create hybrid retriever
        retriever = ingestion_service.create_hybrid_retriever()
        
        # Perform retrieval
        results = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters,
            use_rrf=True
        )
        
        return RetrievalResponse(
            results=results,
            query=request.query,
            total_results=len(results)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval failed: {str(e)}"
        )


@app.post("/ingest/file")
async def ingest_file(file_path: str):
    """
    Ingest a single file into the retrieval system.
    
    Args:
        file_path: Path to the file to ingest
        
    Returns:
        IngestionResult with status
    """
    if ingestion_service is None:
        raise HTTPException(
            status_code=503,
            detail="Retrieval infrastructure not initialized"
        )
    
    try:
        result = ingestion_service.ingest_file(file_path)
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}"
        )


@app.post("/ingest/directory")
async def ingest_directory(
    directory: str,
    recursive: bool = False,
    extensions: Optional[List[str]] = None
):
    """
    Ingest all files from a directory.
    
    Args:
        directory: Path to directory containing documents
        recursive: Whether to search subdirectories
        extensions: File extensions to include (default: all supported)
        
    Returns:
        List of IngestionResult objects
    """
    if ingestion_service is None:
        raise HTTPException(
            status_code=503,
            detail="Retrieval infrastructure not initialized"
        )
    
    try:
        results = ingestion_service.ingest_directory(
            directory=directory,
            recursive=recursive,
            extensions=extensions
        )
        return {"results": results}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Directory ingestion failed: {str(e)}"
        )


@app.get("/ingest/statistics")
async def get_ingestion_statistics():
    """
    Get statistics about ingested documents and indexes.
    
    Returns:
        Dictionary with ingestion and index statistics
    """
    if ingestion_service is None:
        raise HTTPException(
            status_code=503,
            detail="Retrieval infrastructure not initialized"
        )
    
    try:
        stats = ingestion_service.get_statistics()
        return stats
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get statistics: {str(e)}"
        )


@app.delete("/ingest/clear")
async def clear_ingestion_data():
    """
    Clear all ingested data and indexes.
    
    Use with caution - this removes all indexed documents.
    
    Returns:
        Status confirmation
    """
    if ingestion_service is None:
        raise HTTPException(
            status_code=503,
            detail="Retrieval infrastructure not initialized"
        )
    
    try:
        ingestion_service.clear()
        return {"status": "success", "message": "All ingestion data cleared"}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear data: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=DEBUG_MODE
    )
