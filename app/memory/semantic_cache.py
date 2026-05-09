"""
Semantic Cache - Caches query results based on semantic similarity

Avoids redundant LLM calls and retrieval for similar queries.
"""

import hashlib
import time
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from datetime import datetime


class CachedResponse(BaseModel):
    query: str
    response: str
    embedding_hash: str
    created_at: float
    access_count: int = 0
    ttl_seconds: int = 3600  # Default 1 hour
    
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl_seconds


class SemanticCache:
    """
    Semantic cache that matches queries by meaning, not exact text.
    
    Features:
    - Embedding-based similarity matching
    - Configurable similarity threshold
    - TTL-based expiration
    - Access frequency tracking
    - Cache hit/miss metrics
    """
    
    def __init__(
        self, 
        embedding_model=None,
        similarity_threshold: float = 0.95,
        max_size: int = 1000
    ):
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.cache: Dict[str, CachedResponse] = {}
        self.embeddings: Dict[str, List[float]] = {}
        
        # Metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(
        self, 
        query: str, 
        query_embedding: Optional[List[float]] = None
    ) -> Optional[CachedResponse]:
        """
        Retrieve cached response for semantically similar query.
        
        Args:
            query: The query to search for
            query_embedding: Pre-computed embedding (optional)
            
        Returns:
            CachedResponse if found, None otherwise
        """
        if query_embedding is None:
            query_embedding = self._compute_embedding(query)
        
        # Search for similar queries in cache
        best_match = None
        best_similarity = 0.0
        
        for cache_key, cached_item in self.cache.items():
            if cached_item.is_expired():
                self._evict(cache_key)
                continue
            
            # Compute similarity
            cached_embedding = self.embeddings.get(cache_key)
            if cached_embedding:
                similarity = self._cosine_similarity(query_embedding, cached_embedding)
                
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_match = cached_item
        
        if best_match:
            self.hits += 1
            best_match.access_count += 1
            return best_match
        
        self.misses += 1
        return None
    
    def set(
        self, 
        query: str, 
        response: str,
        embedding: Optional[List[float]] = None,
        ttl_seconds: Optional[int] = None
    ) -> str:
        """
        Store response in cache.
        
        Args:
            query: Original query
            response: Response to cache
            embedding: Query embedding (computed if not provided)
            ttl_seconds: Time-to-live in seconds
            
        Returns:
            Cache key
        """
        if embedding is None:
            embedding = self._compute_embedding(query)
        
        # Generate cache key from embedding hash
        cache_key = self._hash_embedding(embedding)
        
        # Enforce max size
        if len(self.cache) >= self.max_size:
            self._evict_least_used()
        
        # Store in cache
        self.cache[cache_key] = CachedResponse(
            query=query,
            response=response,
            embedding_hash=cache_key,
            created_at=time.time(),
            ttl_seconds=ttl_seconds or 3600
        )
        
        self.embeddings[cache_key] = embedding
        
        return cache_key
    
    def _compute_embedding(self, text: str) -> List[float]:
        """Compute embedding for text."""
        if self.embedding_model:
            return self.embedding_model.embed(text)
        
        # Fallback: simple hash-based pseudo-embedding
        # In production, use real embedding model
        hash_bytes = hashlib.sha256(text.encode()).digest()
        # Convert to 256-dimensional vector (normalized)
        embedding = [float(b) / 255.0 for b in hash_bytes]
        # Pad to common dimension (e.g., 1536 for OpenAI)
        embedding += [0.0] * (1536 - len(embedding))
        return embedding
    
    def _hash_embedding(self, embedding: List[float]) -> str:
        """Generate hash from embedding for cache key."""
        # Quantize floats to reduce precision sensitivity
        quantized = [round(x, 2) for x in embedding[:64]]  # Use first 64 dims
        data = ",".join(map(str, quantized))
        return hashlib.md5(data.encode()).hexdigest()
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _evict(self, cache_key: str) -> None:
        """Remove item from cache."""
        if cache_key in self.cache:
            del self.cache[cache_key]
            self.evictions += 1
        
        if cache_key in self.embeddings:
            del self.embeddings[cache_key]
    
    def _evict_least_used(self) -> None:
        """Evict least recently used item."""
        if not self.cache:
            return
        
        # Find item with lowest access count
        min_access = float('inf')
        lru_key = None
        
        for key, item in self.cache.items():
            if item.access_count < min_access:
                min_access = item.access_count
                lru_key = key
        
        if lru_key:
            self._evict(lru_key)
    
    def clear(self) -> None:
        """Clear entire cache."""
        self.cache.clear()
        self.embeddings.clear()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 3),
            "evictions": self.evictions,
            "similarity_threshold": self.similarity_threshold,
        }
