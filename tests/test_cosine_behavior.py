import tempfile
import os
import numpy as np

from paii.db import FaissStore


def test_cosine_normalization_independent_of_scale():
    with tempfile.TemporaryDirectory() as td:
        idx = os.path.join(td, "idx.bin")
        meta = os.path.join(td, "meta.jsonl")

        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        q = np.array([10.0, 0.0, 0.0], dtype=np.float32)

        store_cos = FaissStore(dim=3, index_path=idx + ".c", metadata_path=meta + ".c", metric="cosine")
        store_l2 = FaissStore(dim=3, index_path=idx + ".l", metadata_path=meta + ".l", metric="l2")

        store_cos.add("a", v)
        store_l2.add("a", v)

        res_cos = store_cos.search(q, top_k=1)
        res_l2 = store_l2.search(q, top_k=1)

        assert len(res_cos) == 1
        assert len(res_l2) == 1

        # Cosine should give a near-maximum similarity despite scale
        assert res_cos[0].score > res_l2[0].score
