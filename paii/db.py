"""
FAISS vector database wrapper with atomic saves and JSONL metadata.

Uses IndexIDMap2 for stable id mapping and supports L2 or cosine (via
normalization + IndexFlatIP). Metadata entries include a `text_hash` for
duplicate detection and are written deterministically.
"""

import faiss
import numpy as np
import json
import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import hashlib

from paii.utils import ensure_2d_float32, score_from_distance, normalize_vectors

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: Dict[str, Any]


class FaissStore:
    def __init__(self, dim: int = 384, index_path: str = "data/faiss_index.bin", metadata_path: str = "data/text_data.jsonl", metric: str = "l2"):
        self.dim = dim
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.metric = metric
        self.next_id = 0

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        base = faiss.IndexFlatIP(self.dim) if self.metric == "cosine" else faiss.IndexFlatL2(self.dim)
        self.index = faiss.IndexIDMap2(base)

        self.metadata: Dict[int, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        if self.index_path.exists():
            try:
                loaded = faiss.read_index(str(self.index_path))
                try:
                    _ = loaded.add_with_ids
                    self.index = loaded
                except Exception:
                    self.index = faiss.IndexIDMap2(loaded)
                logger.info(f"Loaded FAISS index from {self.index_path}")
            except Exception as e:
                logger.warning(f"Failed to load FAISS index: {e}. Starting fresh.")
                base = faiss.IndexFlatIP(self.dim) if self.metric == "cosine" else faiss.IndexFlatL2(self.dim)
                self.index = faiss.IndexIDMap2(base)

        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r") as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            entry = json.loads(line.strip())
                            entry_id = entry.get("id")
                            if entry_id is not None:
                                self.metadata[int(entry_id)] = entry
                                self.next_id = max(self.next_id, int(entry_id) + 1)
                        except json.JSONDecodeError as je:
                            logger.warning(f"Skipping malformed JSON at line {line_num}: {je}")
                logger.info(f"Loaded {len(self.metadata)} metadata entries from {self.metadata_path}")
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}. Starting fresh.")

    def _save(self):
        try:
            temp_index = str(self.index_path) + ".tmp"
            faiss.write_index(self.index, temp_index)
            os.replace(temp_index, str(self.index_path))
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            raise

        try:
            temp_metadata = str(self.metadata_path) + ".tmp"
            with open(temp_metadata, "w") as f:
                for _id, entry in sorted(self.metadata.items()):
                    f.write(json.dumps(entry) + "\n")
            os.replace(temp_metadata, str(self.metadata_path))
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            raise

    def add(self, text: str, vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> int:
        vec = ensure_2d_float32(vector)
        if vec.shape[1] != self.dim:
            raise ValueError(f"Vector dimension {vec.shape[1]} does not match store dim {self.dim}")

        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        for entry in self.metadata.values():
            if entry.get("text_hash") == text_hash:
                logger.warning("Duplicate text detected via hash. Skipping.")
                return entry["id"]

        entry_id = self.next_id
        ids = np.array([entry_id], dtype=np.int64)

        vec_to_add = vec
        if self.metric == "cosine":
            vec_to_add = normalize_vectors(vec)

        try:
            self.index.add_with_ids(vec_to_add, ids)
        except Exception:
            self.index.add(vec_to_add)

        self.next_id += 1

        entry_metadata = metadata.copy() if metadata else {}
        entry_metadata.update({
            "id": entry_id,
            "text": text,
            "text_hash": text_hash,
            "created_at": datetime.utcnow().isoformat() + "Z",
        })
        self.metadata[entry_id] = entry_metadata

        self._save()
        logger.info(f"Added entry {entry_id}: {text[:50]}...")
        return entry_id

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[SearchResult]:
        qvec = ensure_2d_float32(query_vector)
        if qvec.shape[1] != self.dim:
            raise ValueError(f"Query vector dimension {qvec.shape[1]} does not match store dim {self.dim}")

        if self.index.ntotal == 0:
            logger.warning("Search on empty index")
            return []

        top_k = min(top_k, self.index.ntotal)

        if self.metric == "cosine":
            qvec = normalize_vectors(qvec)

        distances, indices = self.index.search(qvec, top_k)

        results: List[SearchResult] = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx >= 0 and int(idx) in self.metadata:
                entry = self.metadata[int(idx)]
                score = score_from_distance(distance, metric=self.metric)
                result = SearchResult(
                    text=entry["text"],
                    score=score,
                    metadata={k: v for k, v in entry.items() if k not in ["id", "text", "created_at"]},
                )
                results.append(result)

        results.sort(key=lambda x: x.score, reverse=True)
        logger.info(f"Search returned {len(results)} results")
        return results

    def clear(self):
        logger.warning("Clearing entire store")
        if self.metric == "cosine":
            base = faiss.IndexFlatIP(self.dim)
        else:
            base = faiss.IndexFlatL2(self.dim)
        self.index = faiss.IndexIDMap2(base)
        self.metadata.clear()
        self.next_id = 0
        self._save()

    def size(self) -> int:
        return self.index.ntotal
