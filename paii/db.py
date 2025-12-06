"""
FAISS vector database wrapper with atomic saves and JSONL metadata.
"""

import faiss
import numpy as np
import json
import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from paii.utils import ensure_2d_float32, score_from_distance

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Single search result with text, score, and metadata."""
    text: str
    score: float
    metadata: Dict[str, Any]


class FaissStore:
    """FAISS vector database wrapper with metadata management."""
    
    def __init__(
        self,
        dim: int = 384,
        index_path: str = "data/faiss_index.bin",
        metadata_path: str = "data/text_data.jsonl",
        metric: str = "l2"
    ):
        """
        Initialize FAISS store.
        
        Parameters
        ----------
        dim : int
            Embedding dimension.
        index_path : str
            Path to save/load FAISS index.
        metadata_path : str
            Path to save/load JSONL metadata.
        metric : str
            Distance metric: "l2" or "cosine".
        """
        self.dim = dim
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.metric = metric
        self.next_id = 0
        
        # Create parent directories if needed
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize FAISS index
        if metric == "cosine":
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.index = faiss.IndexFlatL2(dim)
        
        self.metadata = {}  # id -> metadata dict
        
        # Load existing data
        self._load()
    
    def _load(self):
        """Load FAISS index and metadata from disk."""
        # Load index
        if self.index_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                logger.info(f"Loaded FAISS index from {self.index_path}")
            except Exception as e:
                logger.warning(f"Failed to load FAISS index: {e}. Starting fresh.")
                if self.metric == "cosine":
                    self.index = faiss.IndexFlatIP(self.dim)
                else:
                    self.index = faiss.IndexFlatL2(self.dim)
        
        # Load metadata
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r") as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            entry = json.loads(line.strip())
                            entry_id = entry.get("id")
                            if entry_id is not None:
                                self.metadata[entry_id] = entry
                                self.next_id = max(self.next_id, entry_id + 1)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping malformed JSON at line {line_num}: {e}")
                logger.info(f"Loaded {len(self.metadata)} metadata entries from {self.metadata_path}")
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}. Starting fresh.")
    
    def _save(self):
        """Atomically save FAISS index and metadata."""
        # Save index atomically
        try:
            temp_index = str(self.index_path) + ".tmp"
            faiss.write_index(self.index, temp_index)
            os.replace(temp_index, str(self.index_path))
            logger.debug(f"Saved FAISS index to {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            raise
        
        # Save metadata atomically
        try:
            temp_metadata = str(self.metadata_path) + ".tmp"
            with open(temp_metadata, "w") as f:
                for entry in self.metadata.values():
                    f.write(json.dumps(entry) + "\n")
            os.replace(temp_metadata, str(self.metadata_path))
            logger.debug(f"Saved metadata to {self.metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            raise
    
    def add(
        self,
        text: str,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add text and vector to the store.
        
        Parameters
        ----------
        text : str
            Text to store.
        vector : np.ndarray
            Embedding vector (1D or 2D).
        metadata : Optional[Dict[str, Any]]
            Additional metadata (source, page, etc.).
        
        Returns
        -------
        int
            Assigned ID for this entry.
        """
        # Validate and reshape vector
        vector = ensure_2d_float32(vector)
        
        if vector.shape[1] != self.dim:
            raise ValueError(
                f"Vector dimension {vector.shape[1]} does not match store dim {self.dim}"
            )
        
        # Check for duplicates
        for entry in self.metadata.values():
            if entry.get("text") == text:
                logger.warning(f"Duplicate text detected. Skipping: {text[:50]}...")
                return entry["id"]
        
        # Assign ID and add to index
        entry_id = self.next_id
        self.index.add(vector)
        self.next_id += 1
        
        # Store metadata
        entry_metadata = metadata or {}
        entry_metadata.update({
            "id": entry_id,
            "text": text,
            "created_at": datetime.utcnow().isoformat() + "Z",
        })
        self.metadata[entry_id] = entry_metadata
        
        # Persist
        self._save()
        logger.info(f"Added entry {entry_id}: {text[:50]}...")
        
        return entry_id
    
    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[SearchResult]:
        """
        Search the index and return top results.
        
        Parameters
        ----------
        query_vector : np.ndarray
            Query embedding vector (1D or 2D).
        top_k : int
            Number of results to return.
        
        Returns
        -------
        List[SearchResult]
            List of search results sorted by score (descending).
        """
        # Validate and reshape vector
        query_vector = ensure_2d_float32(query_vector)
        
        if query_vector.shape[1] != self.dim:
            raise ValueError(
                f"Query vector dimension {query_vector.shape[1]} does not match store dim {self.dim}"
            )
        
        # Handle empty index
        if self.index.ntotal == 0:
            logger.warning("Search on empty index")
            return []
        
        # Adjust top_k if necessary
        top_k = min(top_k, self.index.ntotal)
        
        # Search
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx >= 0 and idx in self.metadata:
                entry = self.metadata[idx]
                score = score_from_distance(distance, metric=self.metric)
                result = SearchResult(
                    text=entry["text"],
                    score=score,
                    metadata={k: v for k, v in entry.items() if k not in ["id", "text", "created_at"]}
                )
                results.append(result)
        
        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        logger.info(f"Search returned {len(results)} results")
        
        return results
    
    def clear(self):
        """Clear all data from the store."""
        logger.warning("Clearing entire store")
        if self.metric == "cosine":
            self.index = faiss.IndexFlatIP(self.dim)
        else:
            self.index = faiss.IndexFlatL2(self.dim)
        self.metadata.clear()
        self.next_id = 0
        self._save()
    
    def size(self) -> int:
        """Return number of vectors in the index."""
        return self.index.ntotal
