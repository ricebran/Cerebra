"""
Context Assembler - Builds optimal context for LLM generation

Handles:
- Chunk selection and ranking
- Context window optimization
- Deduplication
- Metadata filtering
- Relevance scoring
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class ContextChunk(BaseModel):
    text: str
    score: float
    source: str
    metadata: Dict[str, Any]
    rank: int


class AssembledContext(BaseModel):
    chunks: List[ContextChunk]
    total_tokens: int
    compression_ratio: float
    metadata_summary: Dict[str, Any]


class ContextAssembler:
    """
    Assembles retrieved chunks into optimal context for LLM.
    
    Features:
    - Intelligent chunk selection
    - Token budget management
    - Redundancy elimination
    - Metadata-aware filtering
    - Context compression
    """
    
    def __init__(
        self,
        max_tokens: int = 4000,
        overlap_strategy: str = "merge",
        dedup_threshold: float = 0.9
    ):
        self.max_tokens = max_tokens
        self.overlap_strategy = overlap_strategy
        self.dedup_threshold = dedup_threshold
        self.token_estimator = self._simple_token_count
    
    def assemble(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        query: str,
        max_chunks: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> AssembledContext:
        """
        Assemble retrieved chunks into coherent context.
        
        Args:
            retrieved_chunks: List of retrieved chunks with scores
            query: Original query for relevance checking
            max_chunks: Maximum number of chunks to include
            metadata_filter: Filter chunks by metadata
            
        Returns:
            AssembledContext ready for LLM
        """
        # Filter by metadata if specified
        filtered = self._filter_by_metadata(retrieved_chunks, metadata_filter)
        
        # Sort by relevance score
        sorted_chunks = sorted(filtered, key=lambda x: x.get("score", 0), reverse=True)
        
        # Take top chunks
        if max_chunks:
            sorted_chunks = sorted_chunks[:max_chunks]
        
        # Remove duplicates
        unique_chunks = self._deduplicate(sorted_chunks)
        
        # Build context within token budget
        selected_chunks = self._select_within_budget(unique_chunks)
        
        # Merge overlapping chunks
        merged_chunks = self._merge_overlaps(selected_chunks)
        
        # Calculate metrics
        total_tokens = sum(self.token_estimator(chunk.text) for chunk in merged_chunks)
        original_tokens = sum(self.token_estimator(c["text"]) for c in retrieved_chunks)
        
        return AssembledContext(
            chunks=merged_chunks,
            total_tokens=total_tokens,
            compression_ratio=total_tokens / original_tokens if original_tokens > 0 else 1.0,
            metadata_summary=self._summarize_metadata(merged_chunks)
        )
    
    def _filter_by_metadata(
        self, 
        chunks: List[Dict[str, Any]], 
        filters: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter chunks based on metadata criteria."""
        if not filters:
            return chunks
        
        filtered = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            match = True
            
            for key, value in filters.items():
                if isinstance(value, dict):
                    # Handle operators like $gte, $lte, $in
                    if "$gte" in value:
                        if metadata.get(key, 0) < value["$gte"]:
                            match = False
                    if "$lte" in value:
                        if metadata.get(key, float('inf')) > value["$lte"]:
                            match = False
                    if "$in" in value:
                        if metadata.get(key) not in value["$in"]:
                            match = False
                else:
                    # Exact match
                    if metadata.get(key) != value:
                        match = False
            
            if match:
                filtered.append(chunk)
        
        return filtered
    
    def _deduplicate(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate or near-duplicate chunks."""
        if not chunks:
            return []
        
        unique = [chunks[0]]
        
        for chunk in chunks[1:]:
            is_duplicate = False
            
            for existing in unique:
                similarity = self._text_similarity(chunk["text"], existing["text"])
                if similarity >= self.dedup_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(chunk)
        
        return unique
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Compute simple text similarity (Jaccard index)."""
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _select_within_budget(
        self, 
        chunks: List[Dict[str, Any]]
    ) -> List[ContextChunk]:
        """Select chunks that fit within token budget."""
        selected = []
        current_tokens = 0
        
        for chunk in chunks:
            chunk_tokens = self.token_estimator(chunk["text"])
            
            if current_tokens + chunk_tokens <= self.max_tokens:
                selected.append(ContextChunk(
                    text=chunk["text"],
                    score=chunk.get("score", 0),
                    source=chunk.get("source", "unknown"),
                    metadata=chunk.get("metadata", {}),
                    rank=len(selected)
                ))
                current_tokens += chunk_tokens
            else:
                # Try to truncate chunk if it doesn't fit
                remaining = self.max_tokens - current_tokens
                if remaining > 100:  # Minimum useful chunk size
                    truncated = self._truncate_chunk(chunk, remaining)
                    selected.append(truncated)
                break
        
        return selected
    
    def _truncate_chunk(self, chunk: Dict[str, Any], max_tokens: int) -> ContextChunk:
        """Truncate chunk to fit token limit."""
        text = chunk["text"]
        words = text.split()
        
        # Estimate tokens per word (~0.75 for English)
        estimated_tokens = len(words) * 0.75
        if estimated_tokens <= max_tokens:
            return ContextChunk(
                text=text,
                score=chunk.get("score", 0),
                source=chunk.get("source", "unknown"),
                metadata=chunk.get("metadata", {}),
                rank=0
            )
        
        # Truncate proportionally
        target_words = int(max_tokens / 0.75)
        truncated_text = " ".join(words[:target_words]) + "..."
        
        return ContextChunk(
            text=truncated_text,
            score=chunk.get("score", 0),
            source=chunk.get("source", "unknown"),
            metadata=chunk.get("metadata", {}),
            rank=0
        )
    
    def _merge_overlaps(self, chunks: List[ContextChunk]) -> List[ContextChunk]:
        """Merge overlapping consecutive chunks."""
        if len(chunks) <= 1:
            return chunks
        
        merged = []
        current = chunks[0]
        
        for next_chunk in chunks[1:]:
            # Check if chunks are from same source and consecutive
            if (current.source == next_chunk.source and 
                self._are_consecutive(current, next_chunk)):
                
                # Merge texts
                merged_text = self._merge_texts(current.text, next_chunk.text)
                
                # Keep higher score
                merged_score = max(current.score, next_chunk.score)
                
                current = ContextChunk(
                    text=merged_text,
                    score=merged_score,
                    source=current.source,
                    metadata={**current.metadata, **next_chunk.metadata},
                    rank=current.rank
                )
            else:
                merged.append(current)
                current = next_chunk
        
        merged.append(current)
        return merged
    
    def _are_consecutive(self, chunk1: ContextChunk, chunk2: ContextChunk) -> bool:
        """Check if two chunks are consecutive in source document."""
        meta1 = chunk1.metadata
        meta2 = chunk2.metadata
        
        # Check page numbers or chunk indices
        if "page" in meta1 and "page" in meta2:
            return abs(meta1["page"] - meta2["page"]) <= 1
        
        if "chunk_index" in meta1 and "chunk_index" in meta2:
            return abs(meta1["chunk_index"] - meta2["chunk_index"]) <= 1
        
        return False
    
    def _merge_texts(self, text1: str, text2: str) -> str:
        """Merge two texts, removing overlap."""
        # Simple approach: join with space
        # Advanced: detect and remove overlapping sentences
        words1 = text1.split()
        words2 = text2.split()
        
        # Find overlap at boundary
        overlap_size = min(20, len(words1) // 4)  # Check last 25% of first text
        
        if overlap_size == 0:
            return f"{text1} {text2}"
        
        suffix1 = " ".join(words1[-overlap_size:])
        prefix2 = " ".join(words2[:overlap_size])
        
        # Check for overlap
        if suffix1 in prefix2 or prefix2 in suffix1:
            # Overlap detected, merge intelligently
            return text1 + " " + " ".join(words2[overlap_size:])
        
        return f"{text1} {text2}"
    
    def _summarize_metadata(self, chunks: List[ContextChunk]) -> Dict[str, Any]:
        """Generate summary of metadata across chunks."""
        if not chunks:
            return {}
        
        sources = set(chunk.source for chunk in chunks)
        pages = [chunk.metadata.get("page") for chunk in chunks if "page" in chunk.metadata]
        
        summary = {
            "source_count": len(sources),
            "sources": list(sources)[:5],  # Limit to 5
            "chunk_count": len(chunks),
        }
        
        if pages:
            summary["page_range"] = f"{min(pages)}-{max(pages)}"
        
        return summary
    
    def _simple_token_count(self, text: str) -> int:
        """Simple token count estimator."""
        # Rough estimate: 1 token ≈ 4 characters or 0.75 words
        char_tokens = len(text) / 4
        word_tokens = len(text.split()) * 0.75
        return int(max(char_tokens, word_tokens))
