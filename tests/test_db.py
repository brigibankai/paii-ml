"""
Tests for PAII database module.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path


def test_faiss_store_add_and_search():
    """Test adding and searching in FAISS store."""
    from paii.db import FaissStore
    
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "index.bin"
        metadata_path = Path(tmpdir) / "metadata.jsonl"
        
        # Create store
        store = FaissStore(
            dim=384,
            index_path=str(index_path),
            metadata_path=str(metadata_path),
            metric="l2"
        )
        
        # Add vectors
        vec1 = np.random.randn(1, 384).astype(np.float32)
        vec2 = np.random.randn(1, 384).astype(np.float32)
        
        id1 = store.add("Test text 1", vec1, metadata={"source": "test"})
        id2 = store.add("Test text 2", vec2, metadata={"source": "test"})
        
        assert id1 == 0
        assert id2 == 1
        assert store.size() == 2
        
        # Search
        query = np.random.randn(1, 384).astype(np.float32)
        results = store.search(query, top_k=2)
        
        assert len(results) == 2
        assert all(r.score >= 0 for r in results)


def test_faiss_store_duplicate_detection():
    """Test that duplicates are detected."""
    from paii.db import FaissStore
    
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "index.bin"
        metadata_path = Path(tmpdir) / "metadata.jsonl"
        
        store = FaissStore(
            dim=384,
            index_path=str(index_path),
            metadata_path=str(metadata_path)
        )
        
        vec = np.random.randn(1, 384).astype(np.float32)
        
        # Add first
        id1 = store.add("Duplicate text", vec)
        
        # Try to add same text
        id2 = store.add("Duplicate text", vec)
        
        # Should return same ID, not add twice
        assert id1 == id2
        assert store.size() == 1


def test_faiss_store_persistence():
    """Test saving and loading."""
    from paii.db import FaissStore
    
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "index.bin"
        metadata_path = Path(tmpdir) / "metadata.jsonl"
        
        # Create and populate
        store1 = FaissStore(
            dim=384,
            index_path=str(index_path),
            metadata_path=str(metadata_path)
        )
        
        vec = np.random.randn(1, 384).astype(np.float32)
        store1.add("Test", vec, metadata={"key": "value"})
        
        # Load new instance
        store2 = FaissStore(
            dim=384,
            index_path=str(index_path),
            metadata_path=str(metadata_path)
        )
        
        assert store2.size() == 1
        assert 0 in store2.metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
