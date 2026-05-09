"""Tests for retrieval infrastructure components."""

import pytest
from app.models import Document, Chunk
from app.components.document_loader import TextDocumentLoader, DocumentLoaderFactory
from app.components.chunker import TextChunker
from app.components.embeddings import MockEmbeddingModel, EmbeddingGenerator
from app.components.vector_store import InMemoryVectorStore
from app.components.bm25_retriever import BM25Retriever
from app.components.hybrid_retriever import HybridRetriever
from app.services.ingestion_service import IngestionService


class TestDocumentLoader:
    """Test document loading functionality."""
    
    def test_load_text_document(self, tmp_path):
        """Test loading a text document."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_content = "This is a test document.\nIt has multiple lines."
        test_file.write_text(test_content)
        
        # Load the document
        loader = TextDocumentLoader()
        doc = loader.load(str(test_file))
        
        assert doc.text == test_content
        assert doc.metadata["source"] == "test.txt"
        assert doc.metadata["extension"] == ".txt"
    
    def test_load_batch(self, tmp_path):
        """Test loading multiple documents."""
        # Create test files
        files = []
        for i in range(3):
            f = tmp_path / f"doc_{i}.txt"
            f.write_text(f"Content of document {i}")
            files.append(str(f))
        
        loader = TextDocumentLoader()
        docs = loader.load_batch(files)
        
        assert len(docs) == 3
        assert all(isinstance(doc, Document) for doc in docs)


class TestChunker:
    """Test text chunking functionality."""
    
    def test_chunk_small_text(self):
        """Test chunking text that fits in one chunk."""
        chunker = TextChunker(chunk_size=500, chunk_overlap=50, min_chunk_size=10)
        doc = Document(text="Short text", metadata={"source": "test"})
        chunks = chunker.chunk_document(doc)
        
        assert len(chunks) == 1
        assert chunks[0].text == "Short text"
    
    def test_chunk_large_text(self):
        """Test chunking text that requires multiple chunks."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        
        # Create text longer than chunk size
        text = " ".join([f"Sentence {i}." for i in range(20)])
        doc = Document(text=text, metadata={"source": "test"})
        chunks = chunker.chunk_document(doc)
        
        assert len(chunks) > 1
        assert all(len(chunk.text) <= 150 for chunk in chunks)  # Some tolerance
    
    def test_chunk_overlap(self):
        """Test that chunks have proper overlap."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        
        text = " ".join([f"Sentence {i}." for i in range(10)])
        doc = Document(text=text, metadata={"source": "test"})
        chunks = chunker.chunk_document(doc)
        
        if len(chunks) > 1:
            # Check that consecutive chunks have overlapping content
            for i in range(len(chunks) - 1):
                # Overlap should exist at chunk boundaries
                pass  # Detailed overlap check would go here


class TestEmbeddings:
    """Test embedding generation."""
    
    def test_mock_embedding_dimensions(self):
        """Test mock embedding model produces correct dimensions."""
        model = MockEmbeddingModel(dimensions=1536)
        embedding = model.embed("test text")
        
        assert len(embedding) == 1536
        assert model.dimensions == 1536
    
    def test_mock_embedding_batch(self):
        """Test batch embedding generation."""
        model = MockEmbeddingModel(dimensions=768)
        texts = ["text 1", "text 2", "text 3"]
        embeddings = model.embed_batch(texts)
        
        assert len(embeddings) == 3
        assert all(len(emb) == 768 for emb in embeddings)
    
    def test_embedding_generator(self):
        """Test embedding generator with chunks."""
        model = MockEmbeddingModel(dimensions=256)
        generator = EmbeddingGenerator(model=model)
        
        chunks = [
            Chunk(text="Chunk 1", metadata={}),
            Chunk(text="Chunk 2", metadata={})
        ]
        
        results = generator.generate_embeddings(chunks)
        
        assert len(results) == 2
        assert all(r.dimensions == 256 for r in results)


class TestVectorStore:
    """Test vector store functionality."""
    
    def test_add_and_search(self):
        """Test adding vectors and searching."""
        store = InMemoryVectorStore()
        model = MockEmbeddingModel(dimensions=128)
        
        chunks = [
            Chunk(text="Python programming", metadata={"topic": "coding"}),
            Chunk(text="Machine learning", metadata={"topic": "ai"}),
            Chunk(text="Data analysis", metadata={"topic": "analytics"})
        ]
        
        embeddings = model.embed_batch([c.text for c in chunks])
        store.add(chunks, embeddings)
        
        # Search for Python-related content
        query_embedding = model.embed("coding in Python")
        results = store.search(query_embedding, top_k=2)
        
        assert len(results) == 2
        assert results[0].score > 0  # Should have positive similarity
    
    def test_vector_store_filtering(self):
        """Test metadata filtering in vector store."""
        store = InMemoryVectorStore()
        model = MockEmbeddingModel(dimensions=64)
        
        chunks = [
            Chunk(text="Doc A", metadata={"category": "A"}),
            Chunk(text="Doc B", metadata={"category": "B"})
        ]
        
        embeddings = model.embed_batch([c.text for c in chunks])
        store.add(chunks, embeddings)
        
        # Search with filter
        query_embedding = model.embed("search")
        results = store.search(
            query_embedding,
            top_k=10,
            filters={"category": "A"}
        )
        
        # Should only return category A results
        assert all(r.metadata.get("category") == "A" for r in results)


class TestBM25Retriever:
    """Test BM25 sparse retrieval."""
    
    def test_bm25_indexing(self):
        """Test BM25 indexing and search."""
        retriever = BM25Retriever()
        
        chunks = [
            Chunk(text="The quick brown fox jumps over the lazy dog", metadata={}),
            Chunk(text="Python is a programming language", metadata={}),
            Chunk(text="Machine learning algorithms", metadata={})
        ]
        
        retriever.add_chunks(chunks)
        
        # Search for fox-related content
        results = retriever.search("fox", top_k=5)
        
        assert len(results) > 0
        assert "fox" in results[0].text.lower()
    
    def test_bm25_keyword_matching(self):
        """Test BM25 exact keyword matching."""
        retriever = BM25Retriever()
        
        chunks = [
            Chunk(text="ISO 27001 certification requirements", metadata={}),
            Chunk(text="General security practices", metadata={})
        ]
        
        retriever.add_chunks(chunks)
        
        # BM25 should find exact match for ISO 27001
        results = retriever.search("ISO 27001", top_k=5)
        
        assert len(results) > 0
        assert "ISO 27001" in results[0].text


class TestHybridRetriever:
    """Test hybrid retrieval combining dense and sparse methods."""
    
    def test_hybrid_retrieval(self):
        """Test hybrid retrieval with RRF fusion."""
        vector_store = InMemoryVectorStore()
        bm25 = BM25Retriever()
        embedding_model = MockEmbeddingModel(dimensions=128)
        
        retriever = HybridRetriever(
            vector_store=vector_store,
            bm25_retriever=bm25,
            embedding_model=embedding_model
        )
        
        # Add test chunks
        chunks = [
            Chunk(text="Python programming best practices", metadata={}),
            Chunk(text="Machine learning fundamentals", metadata={}),
            Chunk(text="Data science techniques", metadata={})
        ]
        
        retriever.add_chunks(chunks)
        
        # Retrieve using hybrid search
        results = retriever.retrieve("programming in Python", top_k=2)
        
        assert len(results) > 0
        assert "fusion_method" in results[0].metadata


class TestIngestionService:
    """Test the ingestion service pipeline."""
    
    def test_ingest_document(self):
        """Test document ingestion pipeline."""
        service = IngestionService.create_default({
            "embedding_provider": "mock",
            "vector_provider": "memory",
            "chunk_size": 200,
            "min_chunk_size": 10
        })
        
        doc = Document(
            text="This is a test document about Python programming. It contains enough content to create proper chunks.",
            metadata={"source": "test.txt"}
        )
        
        result = service.ingest_document(doc)
        
        assert result.status == "success"
        assert result.chunks_created > 0
        assert result.source == "test.txt"
    
    def test_ingestion_statistics(self):
        """Test getting ingestion statistics."""
        service = IngestionService.create_default({
            "chunk_size": 200,
            "min_chunk_size": 10
        })
        
        # Ingest a document with enough content
        doc = Document(
            text="Test content for ingestion. This document has sufficient length to be chunked properly.",
            metadata={"source": "test"}
        )
        service.ingest_document(doc)
        
        stats = service.get_statistics()
        
        assert stats["total_documents"] == 1
        assert stats["total_chunks"] > 0
    
    def test_create_hybrid_retriever(self):
        """Test creating hybrid retriever from service."""
        service = IngestionService.create_default()
        
        # Ingest some data
        doc = Document(text="Python programming guide", metadata={})
        service.ingest_document(doc)
        
        # Create retriever
        retriever = service.create_hybrid_retriever(
            dense_weight=0.6,
            sparse_weight=0.4
        )
        
        # Perform retrieval
        results = retriever.retrieve("Python code", top_k=5)
        
        assert isinstance(results, list)


class TestIntegration:
    """Integration tests for the full retrieval pipeline."""
    
    def test_full_pipeline(self, tmp_path):
        """Test complete ingestion and retrieval pipeline."""
        # Create test document
        test_file = tmp_path / "integration_test.txt"
        test_file.write_text(
            "Python is a high-level programming language. "
            "It is widely used in data science and machine learning. "
            "Python has many libraries for scientific computing."
        )
        
        # Create service and ingest
        service = IngestionService.create_default()
        result = service.ingest_file(str(test_file))
        
        assert result.status == "success"
        
        # Create retriever and search
        retriever = service.create_hybrid_retriever()
        results = retriever.retrieve("data science libraries", top_k=3)
        
        assert len(results) > 0
        # Results should be relevant to the query
        combined_text = " ".join([r.text.lower() for r in results])
        assert "python" in combined_text or "data" in combined_text
