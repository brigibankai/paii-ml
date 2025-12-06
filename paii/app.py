"""
PAII System orchestration and main API.
Composes embedding model, vector database, and PDF processor.
"""

import logging
from typing import List, Optional, Dict, Any

from paii.embeddings import EmbeddingModel, EmbeddingFactory
from paii.db import FaissStore, SearchResult
from paii.pdf import PdfProcessor
from paii.utils import ensure_2d_float32
import paii.config as config

logger = logging.getLogger(__name__)


class PAIISystem:
    """Main API for PAII Systems."""
    
    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        vector_db: Optional[FaissStore] = None,
        embedding_provider: str = "local",
        **embedding_kwargs
    ):
        """
        Initialize PAIISystem.
        
        Parameters
        ----------
        embedding_model : Optional[EmbeddingModel]
            Embedding model instance. If None, creates one from provider.
        vector_db : Optional[FaissStore]
            Vector database instance. If None, creates default FAISS store.
        embedding_provider : str
            Provider name if creating embedding model ("local" or "openai").
        **embedding_kwargs
            Additional kwargs for embedding model creation.
        """
        # Create or use provided embedding model
        if embedding_model is None:
            embedding_model = EmbeddingFactory.get(embedding_provider, **embedding_kwargs)
        self.embedding_model = embedding_model
        
        # Create or use provided vector database
        if vector_db is None:
            vector_db = FaissStore(
                dim=embedding_model.dim,
                index_path=str(config.INDEX_PATH),
                metadata_path=str(config.METADATA_PATH),
                metric="l2"
            )
        self.vector_db = vector_db
        
        # Create PDF processor
        self.pdf_processor = PdfProcessor(
            chunk_size=config.DEFAULT_CHUNK_SIZE,
            chunk_overlap=config.DEFAULT_CHUNK_OVERLAP
        )
        
        logger.info(
            f"Initialized PAIISystem with "
            f"embedding_dim={embedding_model.dim}, "
            f"index_size={vector_db.size()}"
        )
    
    def add(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add a single text entry to the vector database.
        
        Parameters
        ----------
        text : str
            Text to add.
        metadata : Optional[Dict[str, Any]]
            Additional metadata (source, page, etc.).
        
        Returns
        -------
        int
            ID assigned to this entry.
        """
        logger.info(f"Adding text to database: {text[:50]}...")
        
        # Embed the text
        embeddings = self.embedding_model.embed([text])
        embeddings = ensure_2d_float32(embeddings)
        
        # Add to vector store
        entry_id = self.vector_db.add(text, embeddings, metadata=metadata)
        return entry_id
    
    def add_bulk(
        self,
        texts: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> List[int]:
        """
        Add multiple texts to the vector database.
        
        Parameters
        ----------
        texts : List[str]
            Texts to add.
        metadata_list : Optional[List[Dict[str, Any]]]
            List of metadata dicts (one per text).
        
        Returns
        -------
        List[int]
            IDs assigned to each entry.
        """
        if metadata_list is None:
            metadata_list = [None] * len(texts)
        
        logger.info(f"Adding {len(texts)} texts to database")
        
        # Embed all texts at once
        embeddings = self.embedding_model.embed(texts)
        embeddings = ensure_2d_float32(embeddings)
        
        ids = []
        for text, emb, metadata in zip(texts, embeddings, metadata_list):
            entry_id = self.vector_db.add(text, emb.reshape(1, -1), metadata=metadata)
            ids.append(entry_id)
        
        return ids
    
    def ingest_pdf(self, pdf_path: str, source_name: Optional[str] = None) -> int:
        """
        Ingest a PDF file: extract, chunk, embed, and store.
        
        Parameters
        ----------
        pdf_path : str
            Path to PDF file.
        source_name : Optional[str]
            Human-readable name for the source.
        
        Returns
        -------
        int
            Number of chunks added.
        """
        logger.info(f"Ingesting PDF: {pdf_path}")
        
        # Extract and chunk
        chunks = self.pdf_processor.process(pdf_path, source_name=source_name)
        
        # Embed and add
        texts = [chunk["text"] for chunk in chunks]
        self.add_bulk(texts, metadata_list=chunks)
        
        logger.info(f"Ingested {len(chunks)} chunks from PDF")
        return len(chunks)
    
    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """
        Search for similar texts in the database.
        
        Parameters
        ----------
        query : str
            Query text.
        top_k : int
            Number of results to return.
        
        Returns
        -------
        List[SearchResult]
            List of search results.
        """
        logger.info(f"Searching for: {query[:50]}...")
        
        # Embed the query
        query_embeddings = self.embedding_model.embed([query])
        query_embeddings = ensure_2d_float32(query_embeddings)
        
        # Search
        results = self.vector_db.search(query_embeddings, top_k=top_k)
        
        logger.info(f"Search returned {len(results)} results")
        return results
    
    def info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "embedding_dim": self.embedding_model.dim,
            "index_size": self.vector_db.size(),
            "index_path": str(self.vector_db.index_path),
            "metadata_path": str(self.vector_db.metadata_path),
        }
