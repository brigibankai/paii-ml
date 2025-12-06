"""
Tests for PAII embeddings module.
"""

import pytest
import numpy as np


def test_ensure_2d_float32():
    """Test dtype and shape validation."""
    from paii.utils import ensure_2d_float32
    
    # 1D array
    arr_1d = np.array([1.0, 2.0, 3.0])
    result = ensure_2d_float32(arr_1d)
    assert result.ndim == 2
    assert result.shape == (1, 3)
    assert result.dtype == np.float32
    
    # 2D array
    arr_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = ensure_2d_float32(arr_2d)
    assert result.ndim == 2
    assert result.shape == (2, 2)
    assert result.dtype == np.float32
    
    # Integer array
    arr_int = np.array([1, 2, 3], dtype=np.int32)
    result = ensure_2d_float32(arr_int)
    assert result.dtype == np.float32


def test_clean_text():
    """Test text cleaning."""
    from paii.utils import clean_text
    
    # Extra spaces
    text = "Hello   world   with    spaces"
    result = clean_text(text)
    assert result == "Hello world with spaces"
    
    # Newlines
    text = "Hello\n\nworld\n"
    result = clean_text(text)
    assert "\n" not in result


def test_chunk_by_length():
    """Test text chunking by length."""
    from paii.utils import chunk_by_length
    
    text = "a" * 1000
    chunks = chunk_by_length(text, chunk_size=300, overlap=50)
    
    assert len(chunks) > 1
    assert all(len(c) <= 350 for c in chunks)  # Some may exceed due to overlap


def test_score_from_distance():
    """Test distance to score conversion."""
    from paii.utils import score_from_distance
    
    # L2: small distance -> high score
    score = score_from_distance(0.1, metric="l2")
    assert 0.8 < score <= 1.0
    
    # L2: large distance -> low score
    score = score_from_distance(10.0, metric="l2")
    assert 0.0 <= score < 0.1
    
    # Cosine: already in [0, 1]
    score = score_from_distance(0.95, metric="cosine")
    assert score == 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
