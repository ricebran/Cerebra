"""Document loader component for loading and processing documents from various sources."""

import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

from app.models import Document


class BaseDocumentLoader(ABC):
    """Abstract base class for document loaders."""
    
    @abstractmethod
    def load(self, source: str) -> Document:
        """Load a document from the given source."""
        pass
    
    @abstractmethod
    def load_batch(self, sources: List[str]) -> List[Document]:
        """Load multiple documents from the given sources."""
        pass


class TextDocumentLoader(BaseDocumentLoader):
    """Loader for plain text files."""
    
    def load(self, source: str, encoding: str = "utf-8") -> Document:
        """Load a text file from disk."""
        path = Path(source)
        
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {source}")
        
        with open(path, "r", encoding=encoding) as f:
            text = f.read()
        
        metadata = {
            "source": path.name,
            "path": str(path.absolute()),
            "size_bytes": path.stat().st_size,
            "extension": path.suffix
        }
        
        return Document(text=text, metadata=metadata)
    
    def load_batch(self, sources: List[str], encoding: str = "utf-8") -> List[Document]:
        """Load multiple text files."""
        documents = []
        for source in sources:
            try:
                doc = self.load(source, encoding)
                documents.append(doc)
            except Exception as e:
                # Log error but continue with other documents
                print(f"Error loading {source}: {e}")
        return documents


class JSONDocumentLoader(BaseDocumentLoader):
    """Loader for JSON files with text extraction."""
    
    def load(self, source: str, text_key: str = "text") -> Document:
        """Load a JSON file and extract text content."""
        path = Path(source)
        
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {source}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Extract text - could be nested or at root level
        if isinstance(data, dict):
            text = data.get(text_key, json.dumps(data))
            metadata = {**data, "source": path.name}
            metadata.pop(text_key, None)
        else:
            text = json.dumps(data)
            metadata = {}
        
        metadata.update({
            "source": path.name,
            "path": str(path.absolute()),
            "size_bytes": path.stat().st_size,
            "extension": ".json"
        })
        
        return Document(text=text, metadata=metadata)
    
    def load_batch(self, sources: List[str], text_key: str = "text") -> List[Document]:
        """Load multiple JSON files."""
        documents = []
        for source in sources:
            try:
                doc = self.load(source, text_key)
                documents.append(doc)
            except Exception as e:
                print(f"Error loading {source}: {e}")
        return documents


class MarkdownDocumentLoader(BaseDocumentLoader):
    """Loader for Markdown files."""
    
    def load(self, source: str, encoding: str = "utf-8") -> Document:
        """Load a Markdown file."""
        path = Path(source)
        
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {source}")
        
        with open(path, "r", encoding=encoding) as f:
            text = f.read()
        
        # Extract title from first heading if present
        metadata = {
            "source": path.name,
            "path": str(path.absolute()),
            "size_bytes": path.stat().st_size,
            "extension": ".md"
        }
        
        lines = text.split("\n")
        for line in lines:
            if line.startswith("# "):
                metadata["title"] = line[2:].strip()
                break
        
        return Document(text=text, metadata=metadata)
    
    def load_batch(self, sources: List[str], encoding: str = "utf-8") -> List[Document]:
        """Load multiple Markdown files."""
        documents = []
        for source in sources:
            try:
                doc = self.load(source, encoding)
                documents.append(doc)
            except Exception as e:
                print(f"Error loading {source}: {e}")
        return documents


class DocumentLoaderFactory:
    """Factory for creating appropriate document loaders based on file type."""
    
    LOADERS = {
        ".txt": TextDocumentLoader,
        ".json": JSONDocumentLoader,
        ".md": MarkdownDocumentLoader,
        ".markdown": MarkdownDocumentLoader,
    }
    
    @classmethod
    def get_loader(cls, source: str) -> BaseDocumentLoader:
        """Get the appropriate loader for the given file source."""
        path = Path(source)
        extension = path.suffix.lower()
        
        loader_class = cls.LOADERS.get(extension)
        if loader_class is None:
            # Default to text loader for unknown extensions
            return TextDocumentLoader()
        
        return loader_class()
    
    @classmethod
    def load(cls, source: str, **kwargs) -> Document:
        """Load a document using the appropriate loader."""
        loader = cls.get_loader(source)
        return loader.load(source, **kwargs)
    
    @classmethod
    def load_batch(cls, sources: List[str], **kwargs) -> List[Document]:
        """Load multiple documents using appropriate loaders."""
        documents = []
        for source in sources:
            try:
                loader = cls.get_loader(source)
                doc = loader.load(source, **kwargs)
                documents.append(doc)
            except Exception as e:
                print(f"Error loading {source}: {e}")
        return documents


def load_documents_from_directory(
    directory: str,
    extensions: Optional[List[str]] = None,
    recursive: bool = False
) -> List[Document]:
    """
    Load all documents from a directory.
    
    Args:
        directory: Path to the directory containing documents
        extensions: List of file extensions to include (e.g., [".txt", ".md"])
                   If None, loads all supported extensions
        recursive: Whether to search subdirectories
    
    Returns:
        List of loaded Document objects
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if extensions is None:
        extensions = list(DocumentLoaderFactory.LOADERS.keys())
    
    documents = []
    
    if recursive:
        files = dir_path.rglob("*")
    else:
        files = dir_path.glob("*")
    
    for file_path in files:
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            try:
                doc = DocumentLoaderFactory.load(str(file_path))
                documents.append(doc)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    return documents
