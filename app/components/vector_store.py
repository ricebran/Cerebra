"""Vector store component for storing and retrieving embedding vectors."""

import uuid
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod

from app.models import Chunk, RetrievalResult


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """Add vectors to the store."""
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> bool:
        """Delete vectors by ID."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all vectors from the store."""
        pass


class InMemoryVectorStore(BaseVectorStore):
    """
    In-memory vector store for testing and development.
    
    Uses cosine similarity for vector search.
    Not suitable for production use with large datasets.
    """
    
    def __init__(self):
        """Initialize in-memory vector store."""
        # Storage: id -> {embedding, chunk}
        self.vectors: Dict[str, Dict[str, Any]] = {}
        
        # Index for metadata filtering
        self.metadata_index: Dict[str, Dict[str, Any]] = {}
    
    def add(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """
        Add vectors to the store.
        
        Args:
            chunks: List of Chunk objects
            embeddings: List of embedding vectors (must match chunks length)
        """
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = chunk.chunk_id or str(uuid.uuid4())
            
            self.vectors[chunk_id] = {
                "embedding": embedding,
                "chunk": chunk
            }
            
            # Index metadata for filtering
            self.metadata_index[chunk_id] = chunk.metadata
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Search for similar vectors using cosine similarity.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of RetrievalResult objects sorted by similarity score
        """
        if not self.vectors:
            return []
        
        # Calculate cosine similarity for all vectors
        scores = []
        
        for chunk_id, data in self.vectors.items():
            # Apply filters if provided
            if filters and not self._matches_filters(chunk_id, filters):
                continue
            
            stored_embedding = data["embedding"]
            chunk = data["chunk"]
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, stored_embedding)
            
            scores.append((chunk_id, similarity, chunk))
        
        # Sort by similarity (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k results
        top_results = scores[:top_k]
        
        # Convert to RetrievalResult objects
        results = []
        for chunk_id, score, chunk in top_results:
            results.append(RetrievalResult(
                text=chunk.text,
                score=score,
                source=chunk.metadata.get("source", "unknown"),
                metadata=chunk.metadata,
                chunk_id=chunk_id
            ))
        
        return results
    
    def delete(self, ids: List[str]) -> bool:
        """
        Delete vectors by ID.
        
        Args:
            ids: List of vector IDs to delete
            
        Returns:
            True if all deletions succeeded
        """
        success = True
        for vector_id in ids:
            if vector_id in self.vectors:
                del self.vectors[vector_id]
                if vector_id in self.metadata_index:
                    del self.metadata_index[vector_id]
            else:
                success = False
        return success
    
    def clear(self) -> None:
        """Clear all vectors from the store."""
        self.vectors.clear()
        self.metadata_index.clear()
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (range: -1 to 1)
        """
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")
        
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Calculate magnitudes
        magnitude1 = sum(a ** 2 for a in vec1) ** 0.5
        magnitude2 = sum(b ** 2 for b in vec2) ** 0.5
        
        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _matches_filters(self, chunk_id: str, filters: Dict[str, Any]) -> bool:
        """
        Check if a chunk matches the given filters.
        
        Args:
            chunk_id: ID of chunk to check
            filters: Filter criteria (key-value pairs)
            
        Returns:
            True if chunk matches all filters
        """
        if chunk_id not in self.metadata_index:
            return False
        
        metadata = self.metadata_index[chunk_id]
        
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            # Handle different filter types
            if isinstance(value, list):
                # List values: match if metadata value is in the list
                if metadata[key] not in value:
                    return False
            elif isinstance(value, dict):
                # Dict values: support operators like $gte, $lte, $in
                if not self._apply_operator_filter(metadata[key], value):
                    return False
            else:
                # Exact match
                if metadata[key] != value:
                    return False
        
        return True
    
    def _apply_operator_filter(self, field_value: Any, operator_dict: Dict[str, Any]) -> bool:
        """
        Apply operator-based filters (e.g., $gte, $lte, $in).
        
        Args:
            field_value: Actual field value from metadata
            operator_dict: Dictionary with operators as keys
            
        Returns:
            True if all operator conditions are satisfied
        """
        for operator, target_value in operator_dict.items():
            if operator == "$gte":
                if field_value < target_value:
                    return False
            elif operator == "$lte":
                if field_value > target_value:
                    return False
            elif operator == "$gt":
                if field_value <= target_value:
                    return False
            elif operator == "$lt":
                if field_value >= target_value:
                    return False
            elif operator == "$ne":
                if field_value == target_value:
                    return False
            elif operator == "$in":
                if field_value not in target_value:
                    return False
            elif operator == "$nin":
                if field_value in target_value:
                    return False
            elif operator == "$exists":
                exists = target_value
                if exists and field_value is None:
                    return False
                if not exists and field_value is not None:
                    return False
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get store statistics.
        
        Returns:
            Dictionary with store statistics
        """
        return {
            "num_vectors": len(self.vectors),
            "dimensions": next(iter(self.vectors.values()))["embedding"].__len__() if self.vectors else 0
        }


class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant vector store for production use.
    
    Qdrant is a high-performance vector database with filtering support.
    Requires Qdrant server running.
    """
    
    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "documents",
        api_key: Optional[str] = None
    ):
        """
        Initialize Qdrant vector store.
        
        Args:
            url: Qdrant server URL
            collection_name: Name of the collection to use
            api_key: API key for authentication (optional)
        """
        self.url = url
        self.collection_name = collection_name
        self.api_key = api_key
        self._client = None
    
    def _get_client(self):
        """Lazy load Qdrant client."""
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError:
                raise ImportError(
                    "qdrant-client not installed. Install with: pip install qdrant-client"
                )
            
            self._client = QdrantClient(
                url=self.url,
                api_key=self.api_key
            )
        
        return self._client
    
    def add(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """Add vectors to Qdrant."""
        client = self._get_client()
        
        # Ensure collection exists
        self._ensure_collection(client, embeddings[0] if embeddings else [])
        
        # Prepare points for upsert
        from qdrant_client.models import PointStruct
        
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = chunk.chunk_id or str(uuid.uuid4())
            
            point = PointStruct(
                id=chunk_id,
                vector=embedding,
                payload={
                    "text": chunk.text,
                    **chunk.metadata
                }
            )
            points.append(point)
        
        client.upsert(collection_name=self.collection_name, points=points)
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Search for similar vectors in Qdrant."""
        client = self._get_client()
        
        # Convert filters to Qdrant format
        qdrant_filter = self._convert_filters(filters) if filters else None
        
        results = client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=qdrant_filter
        )
        
        # Convert to RetrievalResult objects
        retrieval_results = []
        for result in results:
            retrieval_results.append(RetrievalResult(
                text=result.payload.get("text", ""),
                score=result.score,
                source=result.payload.get("source", "unknown"),
                metadata=result.payload,
                chunk_id=str(result.id)
            ))
        
        return retrieval_results
    
    def delete(self, ids: List[str]) -> bool:
        """Delete vectors from Qdrant."""
        client = self._get_client()
        
        from qdrant_client.models import PointIdsList
        
        points_selector = PointIdsList(points=ids)
        result = client.delete(
            collection_name=self.collection_name,
            points_selector=points_selector
        )
        
        return result.status == "completed"
    
    def clear(self) -> None:
        """Clear the collection."""
        client = self._get_client()
        
        # Recreate collection to clear it
        client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config={}
        )
    
    def _ensure_collection(self, client, sample_embedding: List[float]) -> None:
        """Ensure collection exists with proper configuration."""
        from qdrant_client.models import Distance, VectorParams
        
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            dimensions = len(sample_embedding) if sample_embedding else 1536
            
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=dimensions,
                    distance=Cosine
                )
            )
    
    def _convert_filters(self, filters: Dict[str, Any]):
        """Convert filter dict to Qdrant filter format."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        conditions = []
        for key, value in filters.items():
            condition = FieldCondition(
                key=key,
                match=MatchValue(value=value)
            )
            conditions.append(condition)
        
        return Filter(must=conditions) if conditions else None


def create_vector_store(
    provider: str = "memory",
    **kwargs
) -> BaseVectorStore:
    """
    Factory function to create vector stores.
    
    Args:
        provider: Vector store provider ("memory", "qdrant")
        **kwargs: Additional arguments passed to the store constructor
        
    Returns:
        Configured vector store instance
    """
    providers = {
        "memory": InMemoryVectorStore,
        "qdrant": QdrantVectorStore,
    }
    
    store_class = providers.get(provider.lower())
    if store_class is None:
        raise ValueError(
            f"Unknown vector store provider: {provider}. "
            f"Available providers: {list(providers.keys())}"
        )
    
    return store_class(**kwargs)
