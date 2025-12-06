"""
Utility functions for PAII Systems.
Includes chunking, text cleaning, dtype validation, and scoring.
"""

import numpy as np
from typing import List, Tuple
import re


def ensure_2d_float32(array: np.ndarray) -> np.ndarray:
    """
    Ensure array is 2D, float32, and safe for FAISS.
    
    Parameters
    ----------
    array : np.ndarray
        Input array (can be 1D or 2D).
    
    Returns
    -------
    np.ndarray
        2D float32 array with shape (N, dim).
    
    Raises
    ------
    ValueError
        If array is not numeric or conversion fails.
    """
    try:
        array = np.asarray(array, dtype=np.float32)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Cannot convert array to float32: {e}")
    
    if array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got {array.ndim}D")
    
    return np.ascontiguousarray(array)


def clean_text(text: str) -> str:
    """
    Clean text: remove extra whitespace, normalize newlines.
    
    Parameters
    ----------
    text : str
        Raw text.
    
    Returns
    -------
    str
        Cleaned text.
    """
    text = text.strip()
    # Normalize newlines to single space
    text = re.sub(r'\s+', ' ', text)
    return text


def chunk_by_length(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[str]:
    """
    Split text into chunks by character length with overlap.
    
    Parameters
    ----------
    text : str
        Text to chunk.
    chunk_size : int
        Target chunk size in characters.
    overlap : int
        Number of characters to overlap between chunks.
    
    Returns
    -------
    List[str]
        List of chunks.
    """
    text = text.strip()
    if not text:
        return []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move start with overlap
        start = end - overlap if end < len(text) else len(text)
        start = max(start, end - overlap) if chunks else end
    
    return chunks


def chunk_by_paragraphs(
    text: str,
    max_chunk_size: int = 500
) -> List[str]:
    """
    Split text into chunks respecting paragraph boundaries.
    
    Parameters
    ----------
    text : str
        Text to chunk (with \n-separated paragraphs).
    max_chunk_size : int
        Target maximum chunk size in characters.
    
    Returns
    -------
    List[str]
        List of chunks.
    """
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        para_len = len(para)
        if current_length + para_len + 1 > max_chunk_size and current_chunk:
            # Flush current chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = [para]
            current_length = para_len
        else:
            current_chunk.append(para)
            current_length += para_len + 1  # +1 for space
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def score_from_distance(distance: float, metric: str = "l2") -> float:
    """
    Convert FAISS distance to a similarity score in [0, 1].
    
    Parameters
    ----------
    distance : float
        Raw FAISS distance (L2 or IP).
    metric : str
        Metric type: "l2" or "cosine".
    
    Returns
    -------
    float
        Similarity score in [0, 1].
    """
    if metric == "l2":
        # Convert L2 distance to similarity: 1 / (1 + distance)
        score = 1.0 / (1.0 + distance)
    elif metric == "cosine":
        # For cosine (after normalization via IP), distance is already similarity
        score = max(0.0, distance)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return float(np.clip(score, 0.0, 1.0))


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    L2-normalize vectors for cosine similarity via IP.
    
    Parameters
    ----------
    vectors : np.ndarray
        2D array of shape (N, dim).
    
    Returns
    -------
    np.ndarray
        L2-normalized vectors.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # Avoid division by zero
    return vectors / norms
