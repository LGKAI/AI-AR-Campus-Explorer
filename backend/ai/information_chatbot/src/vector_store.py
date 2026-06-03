from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
# HNSW algo : Image it like a Tree. Node (Embedding vectors of chunks).To add a new node, we compare cosine similarity from layer by layer (start at layer 0)
# In layer k, find the most similar node and traverse to descendants of this node (layer k + 1)

class FaissStore:
    """
    FAISS vector store with automatic index-type selection.

    * n < HNSW_THRESHOLD (500): IndexFlatIP — exact brute-force, zero build
      overhead, always correct.  Fine for small corpora (≤ ~500 docs).

    * n ≥ HNSW_THRESHOLD: IndexHNSWFlat with inner-product metric — approximate
      nearest-neighbour, ~5-10x faster at query time on large corpora.  L2-
      normalised embeddings make inner-product == cosine similarity, same
      effective ranking as FlatIP.  efSearch=64 gives ≥ 99 % recall@10 in
      practice for typical RAG corpora.

    Both index types are saved/loaded transparently via faiss.write_index /
    faiss.read_index — no code changes needed after re-ingest.
    """

    # Documents at or above this threshold use HNSWFlat; below uses FlatIP.
    HNSW_THRESHOLD: int = 500

    def __init__(self, index_path: Path, meta_path: Path) -> None:
        self.index_path = index_path
        self.meta_path  = meta_path
        self.index: faiss.Index | None = None
        self.meta:  list[dict[str, Any]] = []

    def build(self, vectors: np.ndarray, meta: list[dict[str, Any]]) -> None:
        dim = vectors.shape[1]
        n   = vectors.shape[0]

        if n >= self.HNSW_THRESHOLD:
            # HNSWFlat with inner-product metric (cosine on normalised vectors)
            index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT) # O(log(n).d)
            index.hnsw.efConstruction = 200   # build-time recall quality
            index.hnsw.efSearch       = 64    # query-time recall/speed balance
        else:
            # Exact brute-force for small corpora
            index = faiss.IndexFlatIP(dim) # O(n.d)

        index.add(vectors)
        self.index = index
        self.meta  = meta

    def save(self) -> None:
        if self.index is None:
            raise ValueError("Index not initialized.")
        faiss.write_index(self.index, str(self.index_path))
        self.meta_path.write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load(self) -> None:
        self.index = faiss.read_index(str(self.index_path))
        self.meta  = json.loads(self.meta_path.read_text(encoding="utf-8"))

    def exists(self) -> bool:
        return self.index_path.exists() and self.meta_path.exists()

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[dict[str, Any]]:
        if self.index is None:
            raise ValueError("Index not loaded.")
        scores, idx = self.index.search(query_vector, k)
        results: list[dict[str, Any]] = []
        for score, i in zip(scores[0], idx[0]):
            if i < 0 or i >= len(self.meta):
                continue
            item = dict(self.meta[i])
            item["score"] = float(score)
            results.append(item)
        return results
