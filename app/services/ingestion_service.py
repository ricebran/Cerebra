"""Ingestion service for processing documents into the retrieval system."""

import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from app.models import Document, Chunk, IngestionResult
from app.components.document_loader import DocumentLoaderFactory, load_documents_from_directory
from app.components.chunker import TextChunker
from app.components.embeddings import EmbeddingGenerator, BaseEmbeddingModel, create_embedding_model
from app.components.vector_store import BaseVectorStore, create_vector_store
from app.components.bm25_retriever import BM25Retriever
from app.components.hybrid_retriever import HybridRetriever


class IngestionService:
    """
    Service for ingesting documents into the retrieval system.
    
    Handles the full pipeline:
    1. Load documents from various sources
    2. Clean and process text
    3. Chunk documents
    4. Generate embeddings
    5. Store in vector database
    6. Index for keyword search
    """
    
    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        vector_store: BaseVectorStore,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        min_chunk_size: int = 50
    ):
        """
        Initialize ingestion service.
        
        Args:
            embedding_model: Model for generating embeddings
            vector_store: Vector store for dense retrieval
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            min_chunk_size: Minimum chunk size to keep
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.embedding_generator = EmbeddingGenerator(model=embedding_model)
        
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size
        )
        
        # BM25 retriever for sparse search
        self.bm25_retriever = BM25Retriever()
        
        # Track ingested documents
        self.ingested_documents: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def create_default(cls, config: Optional[Dict[str, Any]] = None) -> "IngestionService":
        """
        Create ingestion service with default configuration.
        
        Args:
            config: Optional configuration overrides
            
        Returns:
            Configured IngestionService instance
        """
        config = config or {}
        
        # Create embedding model (default: mock for testing)
        embedding_provider = config.get("embedding_provider", "mock")
        embedding_model = create_embedding_model(
            provider=embedding_provider,
            **config.get("embedding_kwargs", {})
        )
        
        # Create vector store (default: in-memory for testing)
        vector_provider = config.get("vector_provider", "memory")
        vector_store = create_vector_store(
            provider=vector_provider,
            **config.get("vector_kwargs", {})
        )
        
        return cls(
            embedding_model=embedding_model,
            vector_store=vector_store,
            chunk_size=config.get("chunk_size", 500),
            chunk_overlap=config.get("chunk_overlap", 100),
            min_chunk_size=config.get("min_chunk_size", 50)
        )
    
    def ingest_document(self, document: Document) -> IngestionResult:
        """
        Ingest a single document into the retrieval system.
        
        Args:
            document: Document object to ingest
            
        Returns:
            IngestionResult with status and metadata
        """
        try:
            source = document.metadata.get("source", "unknown")
            document_id = f"doc_{uuid.uuid4().hex[:12]}"
            
            # Step 1: Chunk the document
            chunks = self.chunker.chunk_document(document)
            
            if not chunks:
                return IngestionResult(
                    document_id=document_id,
                    chunks_created=0,
                    source=source,
                    status="failed",
                    errors=["No chunks created - document may be too short"]
                )
            
            # Step 2: Generate embeddings
            embedding_results = self.embedding_generator.generate_embeddings(chunks)
            embeddings = [result.embedding for result in embedding_results]
            
            # Step 3: Store in vector database
            self.vector_store.add(chunks, embeddings)
            
            # Step 4: Add to BM25 index
            self.bm25_retriever.add_chunks(chunks)
            
            # Track ingestion
            self.ingested_documents[document_id] = {
                "source": source,
                "chunks": len(chunks),
                "ingested_at": datetime.now().isoformat(),
                "metadata": document.metadata
            }
            
            return IngestionResult(
                document_id=document_id,
                chunks_created=len(chunks),
                source=source,
                status="success"
            )
            
        except Exception as e:
            return IngestionResult(
                document_id="unknown",
                chunks_created=0,
                source=document.metadata.get("source", "unknown"),
                status="failed",
                errors=[str(e)]
            )
    
    def ingest_documents(self, documents: List[Document]) -> List[IngestionResult]:
        """
        Ingest multiple documents.
        
        Args:
            documents: List of Document objects to ingest
            
        Returns:
            List of IngestionResult objects
        """
        results = []
        for document in documents:
            result = self.ingest_document(document)
            results.append(result)
        return results
    
    def ingest_file(self, file_path: str) -> IngestionResult:
        """
        Ingest a single file from disk.
        
        Args:
            file_path: Path to the file
            
        Returns:
            IngestionResult with status
        """
        try:
            document = DocumentLoaderFactory.load(file_path)
            return self.ingest_document(document)
        except Exception as e:
            return IngestionResult(
                document_id="unknown",
                chunks_created=0,
                source=file_path,
                status="failed",
                errors=[str(e)]
            )
    
    def ingest_directory(
        self,
        directory: str,
        extensions: Optional[List[str]] = None,
        recursive: bool = False
    ) -> List[IngestionResult]:
        """
        Ingest all documents from a directory.
        
        Args:
            directory: Path to directory containing documents
            extensions: File extensions to include (default: all supported)
            recursive: Whether to search subdirectories
            
        Returns:
            List of IngestionResult objects
        """
        try:
            documents = load_documents_from_directory(
                directory=directory,
                extensions=extensions,
                recursive=recursive
            )
            return self.ingest_documents(documents)
        except Exception as e:
            return [
                IngestionResult(
                    document_id="unknown",
                    chunks_created=0,
                    source=directory,
                    status="failed",
                    errors=[str(e)]
                )
            ]
    
    def create_hybrid_retriever(
        self,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rrf_k: int = 60
    ) -> HybridRetriever:
        """
        Create a hybrid retriever using the indexed data.
        
        Args:
            dense_weight: Weight for dense retrieval
            sparse_weight: Weight for sparse retrieval
            rrf_k: RRF constant
            
        Returns:
            Configured HybridRetriever instance
        """
        return HybridRetriever(
            vector_store=self.vector_store,
            bm25_retriever=self.bm25_retriever,
            embedding_model=self.embedding_model,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            rrf_k=rrf_k
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get ingestion statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_documents": len(self.ingested_documents),
            "total_chunks": sum(
                doc["chunks"] for doc in self.ingested_documents.values()
            ),
            "documents": self.ingested_documents,
            "vector_store_stats": getattr(
                self.vector_store, 'get_statistics', lambda: {}
            )(),
            "bm25_stats": self.bm25_retriever.get_statistics()
        }
    
    def clear(self) -> None:
        """Clear all indexed data."""
        self.vector_store.clear()
        self.bm25_retriever.clear()
        self.ingested_documents.clear()


def ingest_file(
    file_path: str,
    config: Optional[Dict[str, Any]] = None
) -> IngestionResult:
    """
    Convenience function to ingest a single file.
    
    Args:
        file_path: Path to the file
        config: Optional configuration
        
    Returns:
        IngestionResult with status
    """
    service = IngestionService.create_default(config)
    return service.ingest_file(file_path)


def ingest_directory(
    directory: str,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> List[IngestionResult]:
    """
    Convenience function to ingest all files from a directory.
    
    Args:
        directory: Path to directory
        config: Optional configuration
        **kwargs: Additional arguments passed to ingest_directory
        
    Returns:
        List of IngestionResult objects
    """
    service = IngestionService.create_default(config)
    return service.ingest_directory(directory, **kwargs)
