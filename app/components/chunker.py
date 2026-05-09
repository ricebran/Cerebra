"""Text chunking component for splitting documents into retrievable chunks."""

import re
import hashlib
from typing import List, Optional
from app.models import Document, Chunk


class TextChunker:
    """
    Splits text into chunks optimized for retrieval.
    
    Implements recursive chunking with overlap to preserve semantic meaning.
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        min_chunk_size: int = 50
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
            min_chunk_size: Minimum size for a chunk to be kept
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # Validate parameters
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if min_chunk_size > chunk_size:
            raise ValueError("min_chunk_size must be less than or equal to chunk_size")
    
    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """
        Split multiple documents into chunks.
        
        Args:
            documents: List of Document objects to chunk
            
        Returns:
            List of Chunk objects
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Split a single document into chunks.
        
        Args:
            document: Document object to chunk
            
        Returns:
            List of Chunk objects with metadata inherited from document
        """
        text = document.text
        chunks = []
        
        # Use recursive chunking strategy
        chunk_texts = self._recursive_chunk(text)
        
        start_index = 0
        for i, chunk_text in enumerate(chunk_texts):
            if len(chunk_text) < self.min_chunk_size:
                continue
            
            end_index = start_index + len(chunk_text)
            
            # Generate unique chunk ID
            chunk_id = self._generate_chunk_id(
                document.metadata.get("source", "unknown"),
                i,
                chunk_text
            )
            
            # Inherit and augment metadata
            chunk_metadata = {**document.metadata, "chunk_index": i}
            
            chunk = Chunk(
                text=chunk_text,
                metadata=chunk_metadata,
                chunk_id=chunk_id,
                start_index=start_index,
                end_index=end_index
            )
            
            chunks.append(chunk)
            start_index = end_index - self.chunk_overlap
        
        return chunks
    
    def _recursive_chunk(self, text: str, separators: Optional[List[str]] = None) -> List[str]:
        """
        Recursively split text using a list of separators.
        
        Starts with larger semantic units (paragraphs) and falls back to
        smaller units (sentences, words) as needed.
        
        Args:
            text: Text to chunk
            separators: List of separators to use for splitting, in order of preference
            
        Returns:
            List of chunk texts
        """
        if separators is None:
            # Default separators in order of preference
            separators = [
                "\n\n",      # Paragraph breaks
                "\n",        # Line breaks
                ". ",        # Sentence ends
                "! ",        # Exclamation ends
                "? ",        # Question ends
                "; ",        # Clause breaks
                ", ",        # Phrase breaks
                " ",         # Word breaks
                ""           # Character level (last resort)
            ]
        
        # If text fits in one chunk, return it
        if len(text) <= self.chunk_size:
            return [text]
        
        # Try to split using the first separator
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator:
            splits = text.split(separator)
        else:
            # Character-level split
            splits = list(text)
        
        # If only one split result, try next separator
        if len(splits) == 1:
            if remaining_separators:
                return self._recursive_chunk(text, remaining_separators)
            else:
                # Force split by chunk_size if no separators work
                return self._split_by_size(text)
        
        # Process splits
        chunks = []
        current_chunk = ""
        
        for split in splits:
            # Add separator back if it exists
            if separator and split != splits[-1]:
                split_with_sep = split + separator
            else:
                split_with_sep = split
            
            # Check if adding this split exceeds chunk size
            if len(current_chunk) + len(split_with_sep) > self.chunk_size:
                # Save current chunk if it's not empty
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If the split itself is too large, recursively chunk it
                if len(split_with_sep) > self.chunk_size:
                    sub_chunks = self._recursive_chunk(
                        split_with_sep, 
                        remaining_separators
                    )
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = split_with_sep
            else:
                current_chunk += split_with_sep
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Apply overlap
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks)
        
        return chunks
    
    def _split_by_size(self, text: str) -> List[str]:
        """
        Split text into fixed-size chunks without regard for semantics.
        
        Used as a fallback when semantic splitting fails.
        
        Args:
            text: Text to split
            
        Returns:
            List of fixed-size chunk texts
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to break at word boundary
            if end < len(text) and not chunk[-1].isspace():
                last_space = chunk.rfind(" ")
                if last_space > self.min_chunk_size:
                    chunk = chunk[:last_space]
                    end = start + last_space
            
            chunks.append(chunk)
            start = end - self.chunk_overlap
        
        return chunks
    
    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """
        Apply overlap between consecutive chunks.
        
        Args:
            chunks: List of chunk texts without overlap
            
        Returns:
            List of chunk texts with overlap applied
        """
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                # Get overlap from previous chunk
                prev_chunk = overlapped_chunks[-1]
                overlap_start = max(0, len(prev_chunk) - self.chunk_overlap)
                overlap_text = prev_chunk[overlap_start:]
                
                # Prepend overlap to current chunk
                new_chunk = overlap_text + chunk
                overlapped_chunks.append(new_chunk)
        
        return overlapped_chunks
    
    def _generate_chunk_id(self, source: str, index: int, text: str) -> str:
        """
        Generate a unique chunk ID.
        
        Args:
            source: Source document identifier
            index: Chunk index within document
            text: Chunk text
            
        Returns:
            Unique chunk ID string
        """
        # Create hash from source, index, and text content
        content = f"{source}:{index}:{text[:100]}"
        hash_value = hashlib.md5(content.encode()).hexdigest()[:12]
        return f"chunk_{hash_value}"


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[str]:
    """
    Convenience function to chunk text without creating Document objects.
    
    Args:
        text: Text to chunk
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunk texts
    """
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc = Document(text=text, metadata={"source": "inline"})
    chunks = chunker.chunk_document(doc)
    return [chunk.text for chunk in chunks]
