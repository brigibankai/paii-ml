"""
PAII Systems: A lightweight, local-first vector search tool.
"""

__version__ = "0.1.0"

from paii.app import PAIISystem
from paii.embeddings import (
    EmbeddingModel,
    SentenceTransformerEmbedding,
    OpenAIEmbedding,
    EmbeddingFactory,
)
from paii.db import FaissStore, SearchResult

__all__ = [
    "PAIISystem",
    "EmbeddingModel",
    "SentenceTransformerEmbedding",
    "OpenAIEmbedding",
    "EmbeddingFactory",
    "FaissStore",
    "SearchResult",
]
