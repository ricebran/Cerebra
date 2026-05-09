"""BM25 sparse retriever for keyword-based document retrieval."""

import math
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from app.models import Chunk, RetrievalResult


class BM25Retriever:
    """
    BM25 (Best Matching 25) retriever for keyword-based search.
    
    BM25 is a probabilistic ranking function used by search engines.
    It excels at exact keyword matching and complements dense retrieval.
    """
    
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25
    ):
        """
        Initialize BM25 retriever.
        
        Args:
            k1: Term frequency saturation parameter (typically 1.2-2.0)
            b: Length normalization parameter (typically 0.5-0.9)
            epsilon: Score threshold for considering a term relevant
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        
        # Index structures
        self.documents: Dict[str, str] = {}  # doc_id -> text
        self.doc_lengths: Dict[str, int] = {}  # doc_id -> length
        self.avg_doc_length: float = 0.0
        self.num_documents: int = 0
        
        # Inverted index: term -> {doc_id -> frequency}
        self.term_freqs: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Document frequency: term -> number of documents containing term
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        
        # Vocabulary set
        self.vocabulary: set = set()
    
    def add_chunk(self, chunk: Chunk) -> None:
        """
        Add a single chunk to the index.
        
        Args:
            chunk: Chunk object to index
        """
        chunk_id = chunk.chunk_id or f"chunk_{len(self.documents)}"
        text = chunk.text
        
        self.documents[chunk_id] = text
        self.doc_lengths[chunk_id] = len(text.split())
        self.num_documents += 1
        
        # Update average document length
        total_length = sum(self.doc_lengths.values())
        self.avg_doc_length = total_length / self.num_documents
        
        # Tokenize and update inverted index
        tokens = self._tokenize(text)
        for token in tokens:
            self.vocabulary.add(token)
            self.term_freqs[token][chunk_id] += 1
            
            # Update document frequency if this is first occurrence in doc
            if self.term_freqs[token][chunk_id] == 1:
                self.doc_freqs[token] += 1
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Add multiple chunks to the index.
        
        Args:
            chunks: List of Chunk objects to index
        """
        for chunk in chunks:
            self.add_chunk(chunk)
    
    def remove_chunk(self, chunk_id: str) -> bool:
        """
        Remove a chunk from the index.
        
        Args:
            chunk_id: ID of chunk to remove
            
        Returns:
            True if chunk was removed, False if not found
        """
        if chunk_id not in self.documents:
            return False
        
        # Remove from inverted index
        text = self.documents[chunk_id]
        tokens = self._tokenize(text)
        
        for token in tokens:
            if token in self.term_freqs and chunk_id in self.term_freqs[token]:
                del self.term_freqs[token][chunk_id]
                
                # Update document frequency
                if len(self.term_freqs[token]) == 0:
                    del self.doc_freqs[token]
                    del self.vocabulary[token]
                else:
                    self.doc_freqs[token] -= 1
        
        # Remove document records
        del self.documents[chunk_id]
        del self.doc_lengths[chunk_id]
        self.num_documents -= 1
        
        # Recalculate average document length
        if self.num_documents > 0:
            total_length = sum(self.doc_lengths.values())
            self.avg_doc_length = total_length / self.num_documents
        
        return True
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, any]] = None
    ) -> List[RetrievalResult]:
        """
        Search for relevant chunks using BM25 scoring.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of RetrievalResult objects sorted by relevance score
        """
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        # Calculate BM25 scores for all documents
        scores: Dict[str, float] = defaultdict(float)
        
        for token in query_tokens:
            if token not in self.vocabulary:
                continue
            
            # IDF calculation with smoothing
            idf = math.log(
                (self.num_documents - self.doc_freqs[token] + 0.5) /
                (self.doc_freqs[token] + 0.5) + 1.0
            )
            
            # Add to scores for each document containing the term
            for doc_id, freq in self.term_freqs[token].items():
                # Apply filters if provided
                if filters and not self._matches_filters(doc_id, filters):
                    continue
                
                # BM25 formula
                doc_len = self.doc_lengths[doc_id]
                norm_factor = 1.0 - self.b + self.b * (doc_len / self.avg_doc_length)
                
                tf = (freq * (self.k1 + 1.0)) / (freq + self.k1 * norm_factor)
                
                scores[doc_id] += idf * tf
        
        # Sort by score and get top-k
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_results = sorted_results[:top_k]
        
        # Convert to RetrievalResult objects
        results = []
        for doc_id, score in top_results:
            if score < self.epsilon:
                continue
            
            chunk_text = self.documents[doc_id]
            
            # Try to find original chunk metadata
            # In production, you'd store full chunk objects
            results.append(RetrievalResult(
                text=chunk_text,
                score=score,
                source="bm25",
                metadata={"retrieval_method": "bm25"},
                chunk_id=doc_id
            ))
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into terms.
        
        Simple tokenization: lowercase and split on non-alphanumeric.
        Can be extended with stemming, lemmatization, etc.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        import re
        
        # Lowercase and extract alphanumeric tokens
        tokens = re.findall(r'\b[a-z0-9]+\b', text.lower())
        return tokens
    
    def _matches_filters(self, doc_id: str, filters: Dict[str, any]) -> bool:
        """
        Check if a document matches the given filters.
        
        Note: This is a simplified implementation. In production,
        you'd store full metadata for each chunk.
        
        Args:
            doc_id: Document ID to check
            filters: Filter criteria
            
        Returns:
            True if document matches all filters
        """
        # Simplified: always return True since we don't store full metadata
        # In production, implement proper metadata filtering
        return True
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get index statistics.
        
        Returns:
            Dictionary with index statistics
        """
        return {
            "num_documents": self.num_documents,
            "vocabulary_size": len(self.vocabulary),
            "avg_doc_length": self.avg_doc_length,
            "total_terms": sum(self.doc_freqs.values())
        }
    
    def clear(self) -> None:
        """Clear the entire index."""
        self.documents.clear()
        self.doc_lengths.clear()
        self.term_freqs.clear()
        self.doc_freqs.clear()
        self.vocabulary.clear()
        self.num_documents = 0
        self.avg_doc_length = 0.0
