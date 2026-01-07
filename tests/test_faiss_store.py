import tempfile
import numpy as np
import os

from paii.db import FaissStore


def test_add_and_search_basic():
    with tempfile.TemporaryDirectory() as td:
        index_path = os.path.join(td, "faiss.bin")
        meta_path = os.path.join(td, "meta.jsonl")

        store = FaissStore(dim=4, index_path=index_path, metadata_path=meta_path, metric="l2")

        v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        idx = store.add("hello world", v, metadata={"source": "test"})
        assert idx == 0
        assert store.size() == 1

        res = store.search(v, top_k=1)
        assert len(res) == 1
        assert "hello world" in res[0].text
        assert res[0].score > 0


def test_save_and_reload():
    with tempfile.TemporaryDirectory() as td:
        index_path = os.path.join(td, "faiss.bin")
        meta_path = os.path.join(td, "meta.jsonl")

        store = FaissStore(dim=3, index_path=index_path, metadata_path=meta_path, metric="l2")
        v = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        store.add("second", v, metadata={"k": 1})

        # Create a new instance pointing to same files
        store2 = FaissStore(dim=3, index_path=index_path, metadata_path=meta_path, metric="l2")
        assert store2.size() == 1
        res = store2.search(v, top_k=1)
        assert len(res) == 1
        assert "second" in res[0].text
