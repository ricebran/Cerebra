"""
Phase 3: Semantic Caching Service

Caches similar queries to reduce LLM costs and latency.
Uses vector similarity to match incoming queries against cached responses.
"""

import hashlib
import json
import time
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

# Using in-memory store for Phase 3; Redis integration ready for Phase 5
class CacheEntry(BaseModel):
    query_embedding: List[float]
    response: str
    context: Optional[Dict[str, Any]] = None
    created_at: float
    hit_count: int = 0
    ttl: int = 3600  # Time to live in seconds


class SemanticCache:
    """
    Semantic cache that matches queries by embedding similarity.
    """
    
    def __init__(self, similarity_threshold: float = 0.85, max_entries: int = 1000):
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self._cache: Dict[str, CacheEntry] = {}
        self._index: List[str] = []  # Ordered list of keys for LRU
    
    def _compute_key(self, query: str) -> str:
        """Generate deterministic key for exact match fallback."""
        return hashlib.sha256(query.encode()).hexdigest()
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not vec1 or not vec2:
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get(self, query_embedding: List[float]) -> Optional[str]:
        """
        Retrieve cached response if similar query exists.
        
        Args:
            query_embedding: Embedding vector of the incoming query
            
        Returns:
            Cached response string or None if no match
        """
        current_time = time.time()
        
        # Check exact matches first (fast path)
        # In production, this would be a separate hash lookup
        
        best_match_key = None
        best_similarity = 0.0
        
        for key in self._index:
            entry = self._cache.get(key)
            if not entry:
                continue
            
            # Check TTL
            if current_time - entry.created_at > entry.ttl:
                self._remove(key)
                continue
            
            # Compute similarity
            similarity = self._cosine_similarity(query_embedding, entry.query_embedding)
            
            if similarity >= self.similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match_key = key
        
        if best_match_key:
            entry = self._cache[best_match_key]
            entry.hit_count += 1
            # Move to end of index (LRU)
            self._index.remove(best_match_key)
            self._index.append(best_match_key)
            return entry.response
        
        return None
    
    def set(self, query: str, query_embedding: List[float], response: str, 
            context: Optional[Dict[str, Any]] = None, ttl: int = 3600):
        """
        Store query-response pair in cache.
        
        Args:
            query: Original query string
            query_embedding: Embedding vector of the query
            response: Response to cache
            context: Optional metadata (sources, tokens, etc.)
            ttl: Time to live in seconds
        """
        key = self._compute_key(query)
        
        # Evict if at capacity
        if len(self._cache) >= self.max_entries and key not in self._cache:
            self._evict_oldest()
        
        entry = CacheEntry(
            query_embedding=query_embedding,
            response=response,
            context=context,
            created_at=time.time(),
            ttl=ttl
        )
        
        self._cache[key] = entry
        if key not in self._index:
            self._index.append(key)
    
    def _remove(self, key: str):
        """Remove entry from cache."""
        if key in self._cache:
            del self._cache[key]
        if key in self._index:
            self._index.remove(key)
    
    def _evict_oldest(self):
        """Evict oldest entry (LRU)."""
        if self._index:
            oldest_key = self._index.pop(0)
            self._remove(oldest_key)
    
    def clear(self):
        """Clear entire cache."""
        self._cache.clear()
        self._index.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total_hits = sum(e.hit_count for e in self._cache.values())
        return {
            "entries": len(self._cache),
            "max_entries": self.max_entries,
            "total_hits": total_hits,
            "hit_rate": total_hits / max(1, total_hits + len(self._cache)),
            "similarity_threshold": self.similarity_threshold
        }


# Global cache instance
_semantic_cache: Optional[SemanticCache] = None


def get_semantic_cache() -> SemanticCache:
    """Get or create global semantic cache instance."""
    global _semantic_cache
    if _semantic_cache is None:
        _semantic_cache = SemanticCache()
    return _semantic_cache
