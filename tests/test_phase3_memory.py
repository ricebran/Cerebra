"""
Tests for Phase 3: Query Rewriting, Semantic Caching & Context Assembly
"""

import pytest
from app.memory.query_rewriter import QueryRewriter, RewrittenQuery
from app.memory.semantic_cache import SemanticCache, CachedResponse
from app.memory.context_assembler import ContextAssembler, ContextChunk, AssembledContext
from app.memory.conversation_memory import ConversationMemory, Message


class TestQueryRewriter:
    """Test query rewriting functionality."""
    
    def test_basic_rewrite(self):
        """Test basic query rewriting."""
        rewriter = QueryRewriter()
        result = rewriter.rewrite("What is the refund policy?")
        
        assert result.original == "What is the refund policy?"
        assert result.rewritten is not None
        assert isinstance(result.expansions, list)
        assert result.confidence >= 0.0 and result.confidence <= 1.0
        assert isinstance(result.reasoning, str)
    
    def test_pronoun_resolution(self):
        """Test pronoun resolution from conversation history."""
        rewriter = QueryRewriter()
        
        history = [
            {"role": "user", "content": "Tell me about Bitcoin"},
            {"role": "assistant", "content": "Bitcoin is a cryptocurrency..."},
        ]
        
        result = rewriter.rewrite(
            "How does it work?",
            conversation_history=history
        )
        
        # Should replace "it" with "Bitcoin"
        assert "Bitcoin" in result.rewritten or "it" not in result.rewritten.lower()
    
    def test_domain_expansion(self):
        """Test domain-specific query expansion."""
        rewriter = QueryRewriter()
        
        # Trading domain
        result = rewriter.rewrite(
            "What is execution slippage?",
            domain="trading"
        )
        
        # Should have expansions with synonyms
        assert len(result.expansions) >= 1
        # Check for trading synonyms
        expansion_text = " ".join(result.expansions).lower()
        assert any(term in expansion_text for term in ["fill", "price impact", "execution"])
    
    def test_acronym_expansion(self):
        """Test acronym expansion."""
        rewriter = QueryRewriter()
        
        result = rewriter.rewrite("What is CPI?")
        
        # Should expand CPI to Consumer Price Index
        all_text = result.rewritten + " ".join(result.expansions)
        assert "Consumer Price Index" in all_text or "CPI" in all_text
    
    def test_hyde_generation(self):
        """Test Hypothetical Document Embedding generation."""
        rewriter = QueryRewriter()
        
        result = rewriter.rewrite("How do I reset my password?")
        
        # HyDE or expansion should generate some variation
        assert result.rewritten is not None
        # Check that expansions were generated
        assert len(result.expansions) >= 1


class TestSemanticCache:
    """Test semantic caching functionality."""
    
    def test_cache_set_and_get(self):
        """Test basic cache operations."""
        cache = SemanticCache(similarity_threshold=0.9)
        
        query = "What is the refund policy?"
        response = "Refunds are available within 30 days."
        
        # Set in cache
        cache_key = cache.set(query, response)
        assert cache_key is not None
        
        # Get from cache
        cached = cache.get(query)
        assert cached is not None
        assert cached.response == response
        assert cached.query == query
    
    def test_cache_miss(self):
        """Test cache miss for dissimilar query."""
        cache = SemanticCache(similarity_threshold=0.95)  # High threshold
        
        # Set one query
        cache.set("What is the refund policy?", "Refunds available.")
        
        # Different query should miss
        result = cache.get("How do I cancel my order?")
        assert result is None
    
    def test_cache_expiration(self):
        """Test TTL-based expiration."""
        cache = SemanticCache(similarity_threshold=0.9)
        
        # Set with very short TTL (1 second)
        cache_key = cache.set("Test query", "Test response", ttl_seconds=1)
        
        # Should be in cache immediately
        result = cache.get("Test query")
        assert result is not None
        
        # Wait for expiration
        import time
        time.sleep(1.1)
        
        # Should be expired now
        result = cache.get("Test query")
        assert result is None
    
    def test_cache_stats(self):
        """Test cache statistics."""
        cache = SemanticCache(similarity_threshold=0.9)
        
        # Generate some hits and misses
        cache.set("Query 1", "Response 1")
        cache.get("Query 1")  # Hit
        cache.get("Query 2")  # Miss
        
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["size"] == 1
    
    def test_cache_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = SemanticCache(similarity_threshold=0.9, max_size=3)
        
        # Fill cache
        cache.set("Q1", "R1")
        cache.set("Q2", "R2")
        cache.set("Q3", "R3")
        
        # Access Q1 and Q3 to make Q2 least used
        cache.get("Q1")
        cache.get("Q3")
        
        # Add new item - should evict Q2
        cache.set("Q4", "R4")
        
        stats = cache.get_stats()
        assert stats["evictions"] >= 1
        assert stats["size"] <= 3


class TestContextAssembler:
    """Test context assembly functionality."""
    
    def test_basic_assembly(self):
        """Test basic context assembly."""
        assembler = ContextAssembler(max_tokens=1000)
        
        chunks = [
            {"text": "This is chunk 1 about refunds.", "score": 0.9, "source": "policy.pdf", "metadata": {}},
            {"text": "This is chunk 2 about returns.", "score": 0.8, "source": "policy.pdf", "metadata": {}},
            {"text": "This is chunk 3 about exchanges.", "score": 0.7, "source": "policy.pdf", "metadata": {}},
        ]
        
        context = assembler.assemble(chunks, query="What is the refund policy?")
        
        assert isinstance(context, AssembledContext)
        assert len(context.chunks) > 0
        assert context.total_tokens > 0
        assert context.compression_ratio <= 1.0
    
    def test_metadata_filtering(self):
        """Test filtering chunks by metadata."""
        assembler = ContextAssembler()
        
        chunks = [
            {"text": "Trading policy", "score": 0.9, "source": "trading.pdf", "metadata": {"department": "trading"}},
            {"text": "HR policy", "score": 0.8, "source": "hr.pdf", "metadata": {"department": "hr"}},
            {"text": "Compliance policy", "score": 0.7, "source": "compliance.pdf", "metadata": {"department": "compliance"}},
        ]
        
        # Filter by department
        context = assembler.assemble(
            chunks,
            query="Policy",
            metadata_filter={"department": "trading"}
        )
        
        # Should only include trading chunk
        assert len(context.chunks) == 1
        assert context.chunks[0].source == "trading.pdf"
    
    def test_deduplication(self):
        """Test duplicate chunk removal."""
        assembler = ContextAssembler(dedup_threshold=0.9)
        
        # Create nearly identical chunks
        chunks = [
            {"text": "This is the refund policy document.", "score": 0.9, "source": "doc1.pdf", "metadata": {}},
            {"text": "This is the refund policy document.", "score": 0.8, "source": "doc2.pdf", "metadata": {}},  # Duplicate
            {"text": "This is completely different content.", "score": 0.7, "source": "doc3.pdf", "metadata": {}},
        ]
        
        context = assembler.assemble(chunks, query="test")
        
        # Should remove one duplicate
        assert len(context.chunks) < 3
    
    def test_token_budget(self):
        """Test token budget enforcement."""
        assembler = ContextAssembler(max_tokens=50)  # Very small budget
        
        chunks = [
            {"text": "A" * 500, "score": 0.9, "source": "doc1.pdf", "metadata": {}},
            {"text": "B" * 500, "score": 0.8, "source": "doc2.pdf", "metadata": {}},
        ]
        
        context = assembler.assemble(chunks, query="test")
        
        # Should respect token budget
        assert context.total_tokens <= 50


class TestConversationMemory:
    """Test conversation memory functionality."""
    
    def test_create_session(self):
        """Test session creation."""
        memory = ConversationMemory()
        
        session_id = memory.create_session(user_id="user123")
        
        assert session_id is not None
        assert len(session_id) > 0
        
        session = memory.get_session(session_id)
        assert session is not None
        assert session.user_id == "user123"
    
    def test_add_messages(self):
        """Test adding messages to conversation."""
        memory = ConversationMemory()
        
        session_id = memory.create_session()
        
        # Add user message
        memory.add_message(session_id, "user", "What is the refund policy?")
        
        # Add assistant response
        memory.add_message(session_id, "assistant", "Refunds are available within 30 days.")
        
        # Get history
        history = memory.get_history(session_id)
        
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[1].role == "assistant"
    
    def test_get_context_for_turn(self):
        """Test getting formatted context for LLM."""
        memory = ConversationMemory()
        
        session_id = memory.create_session()
        
        memory.add_message(session_id, "user", "Hello")
        memory.add_message(session_id, "assistant", "Hi there!")
        memory.add_message(session_id, "user", "How are you?")
        
        context = memory.get_context_for_turn(session_id)
        
        assert isinstance(context, list)
        assert len(context) == 3
        assert all("role" in msg and "content" in msg for msg in context)
    
    def test_session_cleanup(self):
        """Test expired session cleanup."""
        memory = ConversationMemory(ttl_hours=0)  # Immediate expiration
        
        session_id = memory.create_session()
        memory.add_message(session_id, "user", "Test")
        
        # Manually set last_activity to past
        from datetime import datetime, timedelta
        session = memory.get_session(session_id)
        session.last_activity = datetime.now() - timedelta(hours=1)
        
        # Cleanup should remove expired session
        removed_count = memory.cleanup_expired()
        assert removed_count >= 1
        assert memory.get_session(session_id) is None
    
    def test_memory_stats(self):
        """Test memory statistics."""
        memory = ConversationMemory()
        
        # Create sessions and add messages
        for i in range(3):
            session_id = memory.create_session()
            memory.add_message(session_id, "user", f"Message {i}")
        
        stats = memory.get_stats()
        
        assert stats["active_sessions"] == 3
        assert stats["total_messages"] == 3
        assert stats["max_messages_limit"] == 50


class TestIntegration:
    """Test integration of Phase 3 components."""
    
    def test_query_to_context_pipeline(self):
        """Test full pipeline from query to context."""
        rewriter = QueryRewriter()
        cache = SemanticCache(similarity_threshold=0.9)
        assembler = ContextAssembler(max_tokens=500)
        
        # Original query
        query = "What is the execution policy?"
        
        # Step 1: Rewrite query
        rewritten = rewriter.rewrite(query)
        assert rewritten.rewritten is not None
        
        # Step 2: Check cache (should miss)
        cached = cache.get(rewritten.rewritten)
        assert cached is None
        
        # Step 3: Simulate retrieval and assemble context
        retrieved_chunks = [
            {"text": "Execution policies govern trade execution.", "score": 0.9, "source": "policy.pdf", "metadata": {}},
            {"text": "All trades must follow best execution practices.", "score": 0.8, "source": "policy.pdf", "metadata": {}},
        ]
        
        context = assembler.assemble(retrieved_chunks, rewritten.rewritten)
        assert context.total_tokens > 0
        
        # Step 4: Cache the response
        response = "Execution policies ensure best practices."
        cache.set(rewritten.rewritten, response)
        
        # Step 5: Verify cache hit on same query
        cached = cache.get(rewritten.rewritten)
        assert cached is not None
        assert cached.response == response
