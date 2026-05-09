"""Embedding generation component for converting text to vector representations."""

import os
from typing import List, Optional
from abc import ABC, abstractmethod

from app.models import Chunk, EmbeddingResult


class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass
    
    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the dimension size of embeddings."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass


class MockEmbeddingModel(BaseEmbeddingModel):
    """
    Mock embedding model for testing and development.
    
    Generates deterministic pseudo-embeddings based on text content.
    Useful for testing without API costs.
    """
    
    def __init__(self, dimensions: int = 1536):
        """
        Initialize mock embedding model.
        
        Args:
            dimensions: Number of dimensions for embeddings
        """
        self._dimensions = dimensions
    
    def embed(self, text: str) -> List[float]:
        """Generate deterministic pseudo-embedding based on text."""
        import hashlib
        
        # Create deterministic embedding from text hash
        embedding = []
        for i in range(self._dimensions):
            # Hash text with index to get different values per dimension
            hash_input = f"{text}:{i}".encode()
            hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
            
            # Convert to float in range [-1, 1]
            normalized = (hash_value % 10000) / 5000.0 - 1.0
            embedding.append(normalized)
        
        # Normalize the embedding vector
        magnitude = sum(x ** 2 for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return embedding
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed(text) for text in texts]
    
    @property
    def dimensions(self) -> int:
        return self._dimensions
    
    @property
    def model_name(self) -> str:
        return "mock-embedding-model"


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """
    OpenAI embedding model using their API.
    
    Requires OPENAI_API_KEY environment variable.
    """
    
    def __init__(
        self,
        model: str = "text-embedding-3-large",
        dimensions: Optional[int] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize OpenAI embedding model.
        
        Args:
            model: OpenAI embedding model name
            dimensions: Output dimensions (model-dependent)
            api_key: OpenAI API key (falls back to OPENAI_API_KEY env var)
        """
        self.model = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self._api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable."
            )
        
        # Default dimensions based on model
        if dimensions is None:
            if model == "text-embedding-3-large":
                self._dimensions = 3072
            elif model == "text-embedding-3-small":
                self._dimensions = 1536
            else:
                self._dimensions = 1536
        else:
            self._dimensions = dimensions
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )
        
        client = OpenAI(api_key=self._api_key)
        
        response = client.embeddings.create(
            input=text,
            model=self.model,
            dimensions=self._dimensions if self.model.startswith("text-embedding-3") else None
        )
        
        return response.data[0].embedding
    
    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using batching.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call
            
        Returns:
            List of embedding vectors
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )
        
        client = OpenAI(api_key=self._api_key)
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            response = client.embeddings.create(
                input=batch,
                model=self.model,
                dimensions=self._dimensions if self.model.startswith("text-embedding-3") else None
            )
            
            # Sort by index to maintain order (API may return out of order)
            sorted_data = sorted(response.data, key=lambda x: x.index)
            batch_embeddings = [item.embedding for item in sorted_data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    @property
    def dimensions(self) -> int:
        return self._dimensions
    
    @property
    def model_name(self) -> str:
        return self.model


class SentenceTransformerEmbeddingModel(BaseEmbeddingModel):
    """
    Local embedding model using Hugging Face Sentence Transformers.
    
    Good for offline use and avoiding API costs.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize Sentence Transformer model.
        
        Args:
            model_name: Hugging Face model name
        """
        self.model_name_str = model_name
        self._model = None
        self._dimensions = None
    
    def _load_model(self):
        """Lazy load the model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        self._model = SentenceTransformer(self.model_name_str)
        # Get dimensions from model
        self._dimensions = self._model.get_sentence_embedding_dimension()
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding using local model."""
        if self._model is None:
            self._load_model()
        
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if self._model is None:
            self._load_model()
        
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            self._load_model()
        return self._dimensions
    
    @property
    def model_name(self) -> str:
        return self.model_name_str


class EmbeddingGenerator:
    """
    High-level interface for generating embeddings from chunks.
    
    Handles batching, error handling, and result formatting.
    """
    
    def __init__(self, model: BaseEmbeddingModel, batch_size: int = 50):
        """
        Initialize embedding generator.
        
        Args:
            model: Embedding model instance
            batch_size: Number of texts to process per batch
        """
        self.model = model
        self.batch_size = batch_size
    
    def generate_embeddings(self, chunks: List[Chunk]) -> List[EmbeddingResult]:
        """
        Generate embeddings for a list of chunks.
        
        Args:
            chunks: List of Chunk objects
            
        Returns:
            List of EmbeddingResult objects with text and embeddings
        """
        texts = [chunk.text for chunk in chunks]
        embeddings = self.model.embed_batch(texts)
        
        results = []
        for chunk, embedding in zip(chunks, embeddings):
            result = EmbeddingResult(
                text=chunk.text,
                embedding=embedding,
                model=self.model.model_name,
                dimensions=self.model.dimensions
            )
            results.append(result)
        
        return results
    
    def generate_embedding(self, text: str) -> EmbeddingResult:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            EmbeddingResult object
        """
        embedding = self.model.embed(text)
        
        return EmbeddingResult(
            text=text,
            embedding=embedding,
            model=self.model.model_name,
            dimensions=self.model.dimensions
        )


def create_embedding_model(
    provider: str = "mock",
    **kwargs
) -> BaseEmbeddingModel:
    """
    Factory function to create embedding models.
    
    Args:
        provider: Embedding provider ("mock", "openai", "sentence-transformers")
        **kwargs: Additional arguments passed to the model constructor
        
    Returns:
        Configured embedding model instance
    """
    providers = {
        "mock": MockEmbeddingModel,
        "openai": OpenAIEmbeddingModel,
        "sentence-transformers": SentenceTransformerEmbeddingModel,
    }
    
    model_class = providers.get(provider.lower())
    if model_class is None:
        raise ValueError(
            f"Unknown embedding provider: {provider}. "
            f"Available providers: {list(providers.keys())}"
        )
    
    return model_class(**kwargs)
