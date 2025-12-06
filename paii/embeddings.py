"""
Embedding model abstractions and implementations.
Supports local (SentenceTransformer) and remote (OpenAI) providers.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Convert texts to embeddings.
        
        Parameters
        ----------
        texts : List[str]
            List of texts to embed.
        
        Returns
        -------
        np.ndarray
            2D array of shape (len(texts), embedding_dim), dtype float32.
        """
        pass
    
    @property
    @abstractmethod
    def dim(self) -> int:
        """Return embedding dimension."""
        pass


class SentenceTransformerEmbedding(EmbeddingModel):
    """Local embedding using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize SentenceTransformer embedding model.
        
        Parameters
        ----------
        model_name : str
            Name of the SentenceTransformer model to use.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
        
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self._dim = None
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed texts using SentenceTransformer.
        
        Parameters
        ----------
        texts : List[str]
            Texts to embed.
        
        Returns
        -------
        np.ndarray
            2D array of embeddings, shape (len(texts), 384), dtype float32.
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.dim)
        
        logger.debug(f"Embedding {len(texts)} texts with SentenceTransformer")
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.astype(np.float32)
    
    @property
    def dim(self) -> int:
        """Return embedding dimension (384 for all-MiniLM-L6-v2)."""
        if self._dim is None:
            test_embedding = self.model.encode(["test"])
            self._dim = test_embedding.shape[1]
        return self._dim


class OpenAIEmbedding(EmbeddingModel):
    """Remote embedding using OpenAI API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        """
        Initialize OpenAI embedding.
        
        Parameters
        ----------
        api_key : Optional[str]
            OpenAI API key. If None, reads from OPENAI_API_KEY env var.
        model : str
            OpenAI model name (default: text-embedding-3-small).
        
        Raises
        ------
        ValueError
            If API key is not provided or invalid.
        """
        import os
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY env var or pass api_key."
            )
        
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self._dim = None
        logger.info(f"Initialized OpenAI embedding with model: {model}")
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed texts using OpenAI API.
        
        Parameters
        ----------
        texts : List[str]
            Texts to embed (max 2048 per API call).
        
        Returns
        -------
        np.ndarray
            2D array of embeddings, dtype float32.
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.dim)
        
        # OpenAI API limits batch size; chunk if needed
        batch_size = 100
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            logger.debug(f"Calling OpenAI API for batch of {len(batch)} texts")
            
            response = self.client.embeddings.create(
                model=self.model,
                input=batch
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        logger.debug(f"Generated {len(embeddings)} embeddings")
        return embeddings_array
    
    @property
    def dim(self) -> int:
        """Return embedding dimension (1536 for text-embedding-3-small)."""
        if self._dim is None:
            # Cache dimension after first call
            response = self.client.embeddings.create(
                model=self.model,
                input=["test"]
            )
            self._dim = len(response.data[0].embedding)
        return self._dim


class EmbeddingFactory:
    """Factory for creating embedding models by provider name."""
    
    @staticmethod
    def get(provider: str, **kwargs) -> EmbeddingModel:
        """
        Get an embedding model by provider name.
        
        Parameters
        ----------
        provider : str
            Provider name: "local" or "openai".
        **kwargs
            Additional arguments passed to the model constructor.
        
        Returns
        -------
        EmbeddingModel
            An embedding model instance.
        
        Raises
        ------
        ValueError
            If provider is unknown.
        """
        if provider == "local":
            model_name = kwargs.get("model_name", "all-MiniLM-L6-v2")
            logger.info(f"Creating SentenceTransformerEmbedding: {model_name}")
            return SentenceTransformerEmbedding(model_name=model_name)
        
        elif provider == "openai":
            api_key = kwargs.get("api_key")
            model = kwargs.get("model", "text-embedding-3-small")
            logger.info(f"Creating OpenAIEmbedding: {model}")
            return OpenAIEmbedding(api_key=api_key, model=model)
        
        else:
            raise ValueError(
                f"Unknown provider: {provider}. Choose 'local' or 'openai'."
            )
