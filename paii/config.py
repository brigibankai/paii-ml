"""
Configuration and defaults for PAII Systems.
Reads environment variables for overrides.
"""

import os
from pathlib import Path

# Project root (adjust if needed based on your structure)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directory
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# File paths
INDEX_PATH = DATA_DIR / "faiss_index.bin"
METADATA_PATH = DATA_DIR / "text_data.jsonl"

# Embedding defaults
DEFAULT_LOCAL_MODEL = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIM = 384
DEFAULT_PROVIDER = "local"  # "local" or "openai"

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "text-embedding-3-small")

# PDF and chunking
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

# Search
DEFAULT_TOP_K = 3

# Persistence
ATOMIC_WRITE_TEMP_SUFFIX = ".tmp"

# Logging
LOG_LEVEL = os.getenv("MOONLITE_LOG_LEVEL", "INFO")
