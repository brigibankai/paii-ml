import tempfile
import os
import numpy as np

from paii.db import FaissStore


def test_duplicate_add_returns_same_id():
    with tempfile.TemporaryDirectory() as td:
        idx = os.path.join(td, "idx.bin")
        meta = os.path.join(td, "meta.jsonl")

        store = FaissStore(dim=3, index_path=idx, metadata_path=meta, metric="l2")
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        id1 = store.add("dup text", v)
        id2 = store.add("dup text", v)

        assert id1 == id2
        assert store.size() == 1
