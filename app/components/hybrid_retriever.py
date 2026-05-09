"""Hybrid retriever combining dense and sparse retrieval methods."""

from typing import List, Dict, Optional, Any
from collections import defaultdict

from app.models import Chunk, RetrievalResult
from app.components.vector_store import BaseVectorStore
from app.components.bm25_retriever import BM25Retriever
from app.components.embeddings import BaseEmbeddingModel


class HybridRetriever:
    """
    Hybrid retriever combining dense (vector) and sparse (BM25) retrieval.
    
    Uses Reciprocal Rank Fusion (RRF) to merge results from both methods,
    providing better recall than either method alone.
    """
    
    def __init__(
        self,
        vector_store: BaseVectorStore,
        bm25_retriever: BM25Retriever,
        embedding_model: BaseEmbeddingModel,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rrf_k: int = 60
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: Dense vector store for semantic search
            bm25_retriever: BM25 retriever for keyword search
            embedding_model: Model for generating query embeddings
            dense_weight: Weight for dense retrieval scores in final ranking
            sparse_weight: Weight for sparse retrieval scores in final ranking
            rrf_k: Constant for Reciprocal Rank Fusion (typically 60)
        """
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.embedding_model = embedding_model
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Add chunks to both dense and sparse indexes.
        
        Args:
            chunks: List of Chunk objects to index
        """
        # Generate embeddings for dense retrieval
        embeddings = self.embedding_model.embed_batch([chunk.text for chunk in chunks])
        
        # Add to vector store
        self.vector_store.add(chunks, embeddings)
        
        # Add to BM25 index
        self.bm25_retriever.add_chunks(chunks)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        use_rrf: bool = True
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks using hybrid search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional metadata filters
            use_rrf: Whether to use Reciprocal Rank Fusion
            
        Returns:
            List of RetrievalResult objects sorted by relevance
        """
        # Get dense retrieval results
        query_embedding = self.embedding_model.embed(query)
        dense_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Get more candidates for fusion
            filters=filters
        )
        
        # Get sparse retrieval results
        sparse_results = self.bm25_retriever.search(
            query=query,
            top_k=top_k * 2,
            filters=filters
        )
        
        if use_rrf:
            # Use Reciprocal Rank Fusion
            fused_results = self._reciprocal_rank_fusion(
                dense_results=dense_results,
                sparse_results=sparse_results,
                top_k=top_k
            )
        else:
            # Use weighted score combination
            fused_results = self._weighted_fusion(
                dense_results=dense_results,
                sparse_results=sparse_results,
                top_k=top_k
            )
        
        return fused_results
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
        top_k: int
    ) -> List[RetrievalResult]:
        """
        Merge results using Reciprocal Rank Fusion (RRF).
        
        RRF formula: score = 1/(k + rank_dense) + 1/(k + rank_sparse)
        
        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval
            top_k: Number of results to return
            rrf_k: Constant for RRF (default 60)
            
        Returns:
            Fused and ranked results
        """
        # Create rank mappings
        dense_ranks = {result.chunk_id: rank for rank, result in enumerate(dense_results)}
        sparse_ranks = {result.chunk_id: rank for rank, result in enumerate(sparse_results)}
        
        # Combine all unique chunk IDs
        all_chunk_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        
        # Calculate RRF scores
        rrf_scores = {}
        for chunk_id in all_chunk_ids:
            dense_rank = dense_ranks.get(chunk_id, float('inf'))
            sparse_rank = sparse_ranks.get(chunk_id, float('inf'))
            
            # RRF formula with weights
            score = 0.0
            if dense_rank != float('inf'):
                score += self.dense_weight / (self.rrf_k + dense_rank)
            if sparse_rank != float('inf'):
                score += self.sparse_weight / (self.rrf_k + sparse_rank)
            
            rrf_scores[chunk_id] = score
        
        # Create result lookup maps
        dense_map = {result.chunk_id: result for result in dense_results}
        sparse_map = {result.chunk_id: result for result in sparse_results}
        
        # Sort by RRF score
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Build final results
        results = []
        for chunk_id in sorted_chunk_ids[:top_k]:
            # Prefer dense result metadata, fall back to sparse
            if chunk_id in dense_map:
                base_result = dense_map[chunk_id]
            else:
                base_result = sparse_map[chunk_id]
            
            # Create new result with RRF score
            fused_result = RetrievalResult(
                text=base_result.text,
                score=rrf_scores[chunk_id],
                source=base_result.source,
                metadata={
                    **base_result.metadata,
                    "fusion_method": "rrf",
                    "dense_rank": dense_ranks.get(chunk_id),
                    "sparse_rank": sparse_ranks.get(chunk_id)
                },
                chunk_id=chunk_id
            )
            results.append(fused_result)
        
        return results
    
    def _weighted_fusion(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
        top_k: int
    ) -> List[RetrievalResult]:
        """
        Merge results using weighted score combination.
        
        Normalizes scores from both methods and combines them with weights.
        
        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval
            top_k: Number of results to return
            
        Returns:
            Fused and ranked results
        """
        # Normalize scores to 0-1 range
        dense_map = self._normalize_scores(dense_results)
        sparse_map = self._normalize_scores(sparse_results)
        
        # Combine all unique chunk IDs
        all_chunk_ids = set(dense_map.keys()) | set(sparse_map.keys())
        
        # Calculate weighted scores
        combined_scores = {}
        for chunk_id in all_chunk_ids:
            dense_score = dense_map.get(chunk_id, 0.0)
            sparse_score = sparse_map.get(chunk_id, 0.0)
            
            combined_scores[chunk_id] = (
                self.dense_weight * dense_score +
                self.sparse_weight * sparse_score
            )
        
        # Create result lookup maps
        dense_result_map = {result.chunk_id: result for result in dense_results}
        sparse_result_map = {result.chunk_id: result for result in sparse_results}
        
        # Sort by combined score
        sorted_chunk_ids = sorted(
            combined_scores.keys(),
            key=lambda x: combined_scores[x],
            reverse=True
        )
        
        # Build final results
        results = []
        for chunk_id in sorted_chunk_ids[:top_k]:
            # Prefer dense result, fall back to sparse
            if chunk_id in dense_result_map:
                base_result = dense_result_map[chunk_id]
            else:
                base_result = sparse_result_map[chunk_id]
            
            # Create new result with combined score
            fused_result = RetrievalResult(
                text=base_result.text,
                score=combined_scores[chunk_id],
                source=base_result.source,
                metadata={
                    **base_result.metadata,
                    "fusion_method": "weighted",
                    "dense_score": dense_map.get(chunk_id, 0.0),
                    "sparse_score": sparse_map.get(chunk_id, 0.0)
                },
                chunk_id=chunk_id
            )
            results.append(fused_result)
        
        return results
    
    def _normalize_scores(self, results: List[RetrievalResult]) -> Dict[str, float]:
        """
        Normalize scores to 0-1 range using min-max normalization.
        
        Args:
            results: List of retrieval results
            
        Returns:
            Dictionary mapping chunk_id to normalized score
        """
        if not results:
            return {}
        
        scores = [result.score for result in results]
        min_score = min(scores)
        max_score = max(scores)
        
        # Avoid division by zero
        score_range = max_score - min_score
        if score_range == 0:
            return {result.chunk_id: 1.0 for result in results}
        
        normalized = {}
        for result in results:
            normalized[result.chunk_id] = (result.score - min_score) / score_range
        
        return normalized
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get retriever statistics.
        
        Returns:
            Dictionary with statistics from both retrievers
        """
        return {
            "dense_stats": getattr(self.vector_store, 'get_statistics', lambda: {})(),
            "sparse_stats": self.bm25_retriever.get_statistics(),
            "weights": {
                "dense": self.dense_weight,
                "sparse": self.sparse_weight
            },
            "rrf_k": self.rrf_k
        }
